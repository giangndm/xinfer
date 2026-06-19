//! Hybrid token embedding lookup with host BF16 storage and a bounded device cache.

use std::collections::{HashMap, HashSet, VecDeque};

use candle_core::{DType, Device, Result, Tensor};
use parking_lot::Mutex;

/// Default number of token rows retained in the device hot cache.
pub const DEFAULT_HOT_CACHE_ROWS: usize = 32;

/// Per-lookup cache accounting for observability and benchmark reports.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HybridEmbeddingLookupStats {
    pub tokens: usize,
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
}

impl HybridEmbeddingLookupStats {
    pub fn hit_rate(&self) -> f64 {
        if self.tokens == 0 {
            0.0
        } else {
            self.hits as f64 / self.tokens as f64
        }
    }
}

/// Token embedding table backed by CPU BF16 memory plus a small device cache.
#[derive(Debug)]
pub struct HybridEmbedding {
    inner: Mutex<HybridEmbeddingInner>,
}

#[derive(Debug)]
struct HybridEmbeddingInner {
    host_weights: Tensor,
    vocab_size: usize,
    hidden_size: usize,
    hot_cache_rows: usize,
    hot_order: VecDeque<u32>,
    hot_slots: HashMap<u32, usize>,
    hot_tensor: Option<Tensor>,
    hot_device: Option<Device>,
    hot_dirty: bool,
    last_stats: HybridEmbeddingLookupStats,
}

impl HybridEmbedding {
    pub fn from_tensor(weights: &Tensor, hot_cache_rows: usize) -> Result<Self> {
        let dims = weights.dims();
        if dims.len() != 2 {
            candle_core::bail!("embedding weights must be rank 2, got shape {:?}", weights.shape());
        }
        let vocab_size = dims[0];
        let hidden_size = dims[1];
        let host_weights = weights.to_device(&Device::Cpu)?.to_dtype(DType::BF16)?;

        Ok(Self {
            inner: Mutex::new(HybridEmbeddingInner {
                host_weights,
                vocab_size,
                hidden_size,
                hot_cache_rows,
                hot_order: VecDeque::with_capacity(hot_cache_rows),
                hot_slots: HashMap::with_capacity(hot_cache_rows),
                hot_tensor: None,
                hot_device: None,
                hot_dirty: true,
                last_stats: HybridEmbeddingLookupStats::default(),
            }),
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.lookup(input_ids, input_ids.device(), DType::BF16)
    }

    pub fn lookup(&self, input_ids: &Tensor, device: &Device, output_dtype: DType) -> Result<Tensor> {
        self.inner.lock().lookup(input_ids, device, output_dtype)
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.lock().vocab_size
    }

    pub fn hidden_size(&self) -> usize {
        self.inner.lock().hidden_size
    }

    pub fn hot_cache_rows(&self) -> usize {
        self.inner.lock().hot_cache_rows
    }

    pub fn last_stats(&self) -> HybridEmbeddingLookupStats {
        self.inner.lock().last_stats
    }

    pub fn resident_rows(&self) -> usize {
        self.inner.lock().hot_order.len()
    }
}

impl HybridEmbeddingInner {
    fn lookup(&mut self, input_ids: &Tensor, device: &Device, output_dtype: DType) -> Result<Tensor> {
        let ids = input_ids.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u32>()?;
        if ids.is_empty() {
            return Tensor::zeros((0, self.hidden_size), output_dtype, device);
        }

        if self.hot_cache_rows == 0 {
            self.last_stats = HybridEmbeddingLookupStats {
                tokens: ids.len(),
                hits: 0,
                misses: ids.len(),
                evictions: 0,
            };
            return self.lookup_from_host(&ids, device, output_dtype);
        }

        let unique_ids = ids.iter().copied().collect::<HashSet<_>>();
        let mut stats = HybridEmbeddingLookupStats {
            tokens: ids.len(),
            ..HybridEmbeddingLookupStats::default()
        };
        for id in &ids {
            self.touch_hot_row(*id, &mut stats)?;
        }
        self.last_stats = stats;

        if unique_ids.len() > self.hot_cache_rows {
            return self.lookup_from_host(&ids, device, output_dtype);
        }

        self.rebuild_hot_tensor_if_needed(device)?;
        let slot_ids = ids
            .iter()
            .map(|id| {
                self.hot_slots
                    .get(id)
                    .copied()
                    .ok_or_else(|| candle_core::Error::msg(format!("token id {id} missing from hot cache after touch")))
            })
            .collect::<Result<Vec<_>>>()?;
        let slot_ids = slot_ids.into_iter().map(|slot| slot as u32).collect::<Vec<_>>();
        let slot_tensor = Tensor::from_vec(slot_ids, ids.len(), device)?;
        let hot_tensor = self.hot_tensor.as_ref().ok_or_else(|| candle_core::Error::msg("hot cache tensor was not initialized"))?;
        hot_tensor.index_select(&slot_tensor, 0)?.to_dtype(output_dtype)
    }

    fn touch_hot_row(&mut self, id: u32, stats: &mut HybridEmbeddingLookupStats) -> Result<()> {
        if id as usize >= self.vocab_size {
            candle_core::bail!("token id {id} is outside embedding vocab size {}", self.vocab_size);
        }
        if self.hot_slots.contains_key(&id) {
            stats.hits += 1;
            return Ok(());
        }

        stats.misses += 1;
        if self.hot_order.len() == self.hot_cache_rows {
            let evicted = self.hot_order.pop_front().ok_or_else(|| candle_core::Error::msg("hot cache capacity reached with empty order"))?;
            self.hot_slots.remove(&evicted);
            stats.evictions += 1;
        }
        self.hot_order.push_back(id);
        self.reindex_hot_slots();
        self.hot_dirty = true;
        Ok(())
    }

    fn reindex_hot_slots(&mut self) {
        self.hot_slots.clear();
        for (slot, id) in self.hot_order.iter().copied().enumerate() {
            self.hot_slots.insert(id, slot);
        }
    }

    fn rebuild_hot_tensor_if_needed(&mut self, device: &Device) -> Result<()> {
        let hot_device_matches = self.hot_device.as_ref().is_some_and(|hot_device| hot_device.same_device(device));
        if !self.hot_dirty && self.hot_tensor.is_some() && hot_device_matches {
            return Ok(());
        }
        let ids = self.hot_order.iter().copied().collect::<Vec<_>>();
        self.hot_tensor = Some(self.lookup_from_host(&ids, device, DType::BF16)?);
        self.hot_device = Some(device.clone());
        self.hot_dirty = false;
        Ok(())
    }

    fn lookup_from_host(&self, ids: &[u32], device: &Device, output_dtype: DType) -> Result<Tensor> {
        for id in ids {
            if *id as usize >= self.vocab_size {
                candle_core::bail!("token id {id} is outside embedding host storage");
            }
        }
        let id_tensor = Tensor::from_vec(ids.to_vec(), ids.len(), &Device::Cpu)?;
        self.host_weights.index_select(&id_tensor, 0)?.to_device(device)?.to_dtype(output_dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_weights() -> Tensor {
        Tensor::new(&[[0.0f32, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]], &Device::Cpu).unwrap()
    }

    #[test]
    fn stores_cpu_bf16_rows() {
        let embedding = HybridEmbedding::from_tensor(&test_weights(), 2).unwrap();
        let ids = Tensor::new(&[0u32], &Device::Cpu).unwrap();

        assert_eq!(embedding.vocab_size(), 4);
        assert_eq!(embedding.hidden_size(), 3);
        assert_eq!(embedding.lookup(&ids, &Device::Cpu, DType::BF16).unwrap().dtype(), DType::BF16);
    }

    #[test]
    fn default_hot_cache_rows_matches_voice_assistant_budget() {
        assert_eq!(DEFAULT_HOT_CACHE_ROWS, 32);
    }

    #[test]
    fn matches_baseline_lookup() {
        let weights = test_weights();
        let embedding = HybridEmbedding::from_tensor(&weights, 2).unwrap();
        let ids = Tensor::new(&[2u32, 0, 2], &Device::Cpu).unwrap();

        let output = embedding.lookup(&ids, &Device::Cpu, DType::F32).unwrap();
        let baseline = weights.index_select(&ids, 0).unwrap();

        assert_close(&output.to_vec2::<f32>().unwrap(), &baseline.to_vec2::<f32>().unwrap(), 0.01);
        assert_eq!(
            embedding.last_stats(),
            HybridEmbeddingLookupStats {
                tokens: 3,
                hits: 1,
                misses: 2,
                evictions: 0,
            }
        );
    }

    #[test]
    fn tracks_capacity_evictions() {
        let embedding = HybridEmbedding::from_tensor(&test_weights(), 2).unwrap();
        let ids = Tensor::new(&[0u32, 1, 2, 1], &Device::Cpu).unwrap();

        let _ = embedding.lookup(&ids, &Device::Cpu, DType::BF16).unwrap();

        assert_eq!(embedding.resident_rows(), 2);
        assert_eq!(
            embedding.last_stats(),
            HybridEmbeddingLookupStats {
                tokens: 4,
                hits: 1,
                misses: 3,
                evictions: 1,
            }
        );
    }

    #[test]
    fn handles_unique_ids_larger_than_hot_cache() {
        let vocab_size = 4096;
        let hidden_size = 2;
        let values = (0..vocab_size * hidden_size).map(|idx| (idx % 64) as f32 / 64.0).collect::<Vec<_>>();
        let weights = Tensor::from_vec(values, (vocab_size, hidden_size), &Device::Cpu).unwrap();
        let ids = (0..vocab_size as u32).collect::<Vec<_>>();
        let ids_tensor = Tensor::from_vec(ids, vocab_size, &Device::Cpu).unwrap();
        let embedding = HybridEmbedding::from_tensor(&weights, 1024).unwrap();

        let output = embedding.lookup(&ids_tensor, &Device::Cpu, DType::F32).unwrap();
        let baseline = weights.index_select(&ids_tensor, 0).unwrap();

        assert_close(&output.to_vec2::<f32>().unwrap(), &baseline.to_vec2::<f32>().unwrap(), 0.01);
        assert_eq!(
            embedding.last_stats(),
            HybridEmbeddingLookupStats {
                tokens: vocab_size,
                hits: 0,
                misses: vocab_size,
                evictions: vocab_size - 1024,
            }
        );
    }

    #[test]
    fn can_disable_hot_cache() {
        let embedding = HybridEmbedding::from_tensor(&test_weights(), 0).unwrap();
        let ids = Tensor::new(&[0u32, 1, 0], &Device::Cpu).unwrap();

        let output = embedding.lookup(&ids, &Device::Cpu, DType::F32).unwrap();

        assert_eq!(embedding.resident_rows(), 0);
        assert_eq!(output.dims(), &[3, 3]);
        assert_eq!(
            embedding.last_stats(),
            HybridEmbeddingLookupStats {
                tokens: 3,
                hits: 0,
                misses: 3,
                evictions: 0,
            }
        );
    }

    #[test]
    fn rejects_out_of_range_ids() {
        let embedding = HybridEmbedding::from_tensor(&test_weights(), 2).unwrap();
        let ids = Tensor::new(&[0u32, 4], &Device::Cpu).unwrap();

        assert!(embedding.lookup(&ids, &Device::Cpu, DType::F32).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn rebuilds_hot_tensor_when_lookup_device_changes() {
        let cuda = Device::new_cuda(0).unwrap();
        let embedding = HybridEmbedding::from_tensor(&test_weights(), 2).unwrap();
        let cuda_ids = Tensor::new(&[0u32, 1], &cuda).unwrap();

        let _ = embedding.lookup(&cuda_ids, &cuda, DType::BF16).unwrap();

        let cpu_ids = Tensor::new(&[0u32, 1], &Device::Cpu).unwrap();
        let output = embedding.lookup(&cpu_ids, &Device::Cpu, DType::F32).unwrap();
        let baseline = test_weights().index_select(&cpu_ids, 0).unwrap();

        assert!(output.device().same_device(&Device::Cpu));
        assert_close(&output.to_vec2::<f32>().unwrap(), &baseline.to_vec2::<f32>().unwrap(), 0.01);
    }

    fn assert_close(actual: &[Vec<f32>], expected: &[Vec<f32>], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (actual_row, expected_row) in actual.iter().zip(expected) {
            assert_eq!(actual_row.len(), expected_row.len());
            for (actual_value, expected_value) in actual_row.iter().zip(expected_row) {
                assert!(
                    (actual_value - expected_value).abs() <= tolerance,
                    "actual={actual_value} expected={expected_value} tolerance={tolerance}"
                );
            }
        }
    }
}
