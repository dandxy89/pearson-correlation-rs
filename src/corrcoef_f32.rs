use packed_simd::f32x8;

const SIZE: usize = 8;

#[must_use]
/// The Pearson correlation coefficient measures the linear relationship between
/// the two sequences of values.
///
/// It is equivalent to the numpy function `np.corrcoef`.
///
/// # Arguments
/// * `seq1` - The first sequence
/// * `seq2` - The second sequence
///
/// # Panics
pub fn corrcoef_f32x8(seq1: &[f32], seq2: &[f32]) -> f32 {
    let seq_len = seq1.len();
    assert_eq!(seq_len, seq2.len());

    let mut sum_sq1 = f32x8::splat(0.0);
    let mut sum_sq2 = f32x8::splat(0.0);
    let mut dot_product = f32x8::splat(0.0);

    for i in (0..seq_len).step_by(SIZE) {
        if i + SIZE > seq_len {
            break;
        }
        let vec1 = f32x8::from_slice_unaligned(&seq1[i..i + SIZE]);
        let vec2 = f32x8::from_slice_unaligned(&seq2[i..i + SIZE]);

        dot_product += vec1 * vec2;
        sum_sq1 += vec1 * vec1;
        sum_sq2 += vec2 * vec2;
    }

    let mut sum_sq1_sum = sum_sq1.extract(0) + sum_sq1.extract(1);
    let mut sum_sq2_sum = sum_sq2.extract(0) + sum_sq2.extract(1);
    let mut dot_product_sum = dot_product.extract(0) + dot_product.extract(1);

    // Handle remaining elements (if any || seq_len % SIZE != 0)
    for i in (seq_len - seq_len % SIZE)..seq_len {
        dot_product_sum += seq1[i] * seq2[i];
        sum_sq1_sum += seq1[i] * seq1[i];
        sum_sq2_sum += seq2[i] * seq2[i];
    }

    let denom = sum_sq1_sum.sqrt() * sum_sq2_sum.sqrt();
    dot_product_sum / denom
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use super::corrcoef_f32x8;

    #[test]
    fn correlation_basic() {
        let seq1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let seq2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_float_eq!(corrcoef_f32x8(&seq1, &seq2), 1.0, abs <= 1e-10);
    }
}
