//! G.711 codec conversions (ITU-T mu-law and A-law).

// --------------------------------------------------------------------------
// mu-law (G.711 u-law) lookup tables
// --------------------------------------------------------------------------

/// Decode table: mu-law byte -> PCM i16 value.
#[rustfmt::skip]
const ULAW_DECODE_TABLE: [i16; 256] = [
    -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
    -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
    -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
    -11900, -11388, -10876, -10364,  -9852,  -9340,  -8828,  -8316,
     -7932,  -7676,  -7420,  -7164,  -6908,  -6652,  -6396,  -6140,
     -5884,  -5628,  -5372,  -5116,  -4860,  -4604,  -4348,  -4092,
     -3900,  -3772,  -3644,  -3516,  -3388,  -3260,  -3132,  -3004,
     -2876,  -2748,  -2620,  -2492,  -2364,  -2236,  -2108,  -1980,
     -1884,  -1820,  -1756,  -1692,  -1628,  -1564,  -1500,  -1436,
     -1372,  -1308,  -1244,  -1180,  -1116,  -1052,   -988,   -924,
      -876,   -844,   -812,   -780,   -748,   -716,   -684,   -652,
      -620,   -588,   -556,   -524,   -492,   -460,   -428,   -396,
      -372,   -356,   -340,   -324,   -308,   -292,   -276,   -260,
      -244,   -228,   -212,   -196,   -180,   -164,   -148,   -132,
      -120,   -112,   -104,    -96,    -88,    -80,    -72,    -64,
       -56,    -48,    -40,    -32,    -24,    -16,     -8,      0,
     32124,  31100,  30076,  29052,  28028,  27004,  25980,  24956,
     23932,  22908,  21884,  20860,  19836,  18812,  17788,  16764,
     15996,  15484,  14972,  14460,  13948,  13436,  12924,  12412,
     11900,  11388,  10876,  10364,   9852,   9340,   8828,   8316,
      7932,   7676,   7420,   7164,   6908,   6652,   6396,   6140,
      5884,   5628,   5372,   5116,   4860,   4604,   4348,   4092,
      3900,   3772,   3644,   3516,   3388,   3260,   3132,   3004,
      2876,   2748,   2620,   2492,   2364,   2236,   2108,   1980,
      1884,   1820,   1756,   1692,   1628,   1564,   1500,   1436,
      1372,   1308,   1244,   1180,   1116,   1052,    988,    924,
       876,    844,    812,    780,    748,    716,    684,    652,
       620,    588,    556,    524,    492,    460,    428,    396,
       372,    356,    340,    324,    308,    292,    276,    260,
       244,    228,    212,    196,    180,    164,    148,    132,
       120,    112,    104,     96,     88,     80,     72,     64,
        56,     48,     40,     32,     24,     16,      8,      0,
];

/// Decode mu-law encoded bytes to PCM i16 samples.
pub fn ulaw_decode(input: &[u8]) -> Vec<i16> {
    input.iter().map(|&b| ULAW_DECODE_TABLE[b as usize]).collect()
}

/// Encode PCM i16 samples to mu-law bytes.
///
/// Uses the standard ITU-T G.711 algorithm with bias of 0x84 and segment lookup.
pub fn ulaw_encode(input: &[i16]) -> Vec<u8> {
    input.iter().map(|&sample| encode_ulaw_sample(sample)).collect()
}

fn encode_ulaw_sample(sample: i16) -> u8 {
    const BIAS: i32 = 0x84;
    const CLIP: i32 = 32635;

    // Table maps segment number to the mu-law encoded segment bits.
    #[rustfmt::skip]
    const SEG_TABLE: [i32; 8] = [
        0x3F, 0x1F, 0x0F, 0x07, 0x03, 0x01, 0x00, 0x00,
    ];
    let _ = SEG_TABLE; // suppress unused warning — kept for documentation.

    let sign: i32;
    let mut pcm_val = sample as i32;

    // Get the sign and make the value positive.
    if pcm_val < 0 {
        pcm_val = -pcm_val;
        sign = 0x80;
    } else {
        sign = 0;
    }

    if pcm_val > CLIP {
        pcm_val = CLIP;
    }
    pcm_val += BIAS;

    // Find the segment (exponent).
    let mut segment: i32 = 7;
    let mut mask: i32 = 0x4000;
    while segment > 0 {
        if pcm_val & mask != 0 {
            break;
        }
        segment -= 1;
        mask >>= 1;
    }

    // Combine sign, segment, and quantization bits, then invert.
    let uval = (sign | (segment << 4) | ((pcm_val >> (segment + 3)) & 0x0F)) as u8;
    !uval
}

// --------------------------------------------------------------------------
// A-law (G.711 A-law) lookup tables
// --------------------------------------------------------------------------

/// Decode table: A-law byte -> PCM i16 value.
#[rustfmt::skip]
const ALAW_DECODE_TABLE: [i16; 256] = [
     -5504,  -5248,  -6016,  -5760,  -4480,  -4224,  -4992,  -4736,
     -7552,  -7296,  -8064,  -7808,  -6528,  -6272,  -7040,  -6784,
     -2752,  -2624,  -3008,  -2880,  -2240,  -2112,  -2496,  -2368,
     -3776,  -3648,  -4032,  -3904,  -3264,  -3136,  -3520,  -3392,
    -22016, -20992, -24064, -23040, -17920, -16896, -19968, -18944,
    -30208, -29184, -32256, -31232, -26112, -25088, -28160, -27136,
    -11008, -10496, -12032, -11520,  -8960,  -8448,  -9984,  -9472,
    -15104, -14592, -16128, -15616, -13056, -12544, -14080, -13568,
      -344,   -328,   -376,   -360,   -280,   -264,   -312,   -296,
      -472,   -456,   -504,   -488,   -408,   -392,   -440,   -424,
       -88,    -72,   -120,   -104,    -24,     -8,    -56,    -40,
      -216,   -200,   -248,   -232,   -152,   -136,   -184,   -168,
     -1376,  -1312,  -1504,  -1440,  -1120,  -1056,  -1248,  -1184,
     -1888,  -1824,  -2016,  -1952,  -1632,  -1568,  -1760,  -1696,
      -688,   -656,   -752,   -720,   -560,   -528,   -624,   -592,
      -944,   -912,  -1008,   -976,   -816,   -784,   -880,   -848,
      5504,   5248,   6016,   5760,   4480,   4224,   4992,   4736,
      7552,   7296,   8064,   7808,   6528,   6272,   7040,   6784,
      2752,   2624,   3008,   2880,   2240,   2112,   2496,   2368,
      3776,   3648,   4032,   3904,   3264,   3136,   3520,   3392,
     22016,  20992,  24064,  23040,  17920,  16896,  19968,  18944,
     30208,  29184,  32256,  31232,  26112,  25088,  28160,  27136,
     11008,  10496,  12032,  11520,   8960,   8448,   9984,   9472,
     15104,  14592,  16128,  15616,  13056,  12544,  14080,  13568,
       344,    328,    376,    360,    280,    264,    312,    296,
       472,    456,    504,    488,    408,    392,    440,    424,
        88,     72,    120,    104,     24,      8,     56,     40,
       216,    200,    248,    232,    152,    136,    184,    168,
      1376,   1312,   1504,   1440,   1120,   1056,   1248,   1184,
      1888,   1824,   2016,   1952,   1632,   1568,   1760,   1696,
       688,    656,    752,    720,    560,    528,    624,    592,
       944,    912,   1008,    976,    816,    784,    880,    848,
];

/// Decode A-law encoded bytes to PCM i16 samples.
pub fn alaw_decode(input: &[u8]) -> Vec<i16> {
    input.iter().map(|&b| ALAW_DECODE_TABLE[b as usize]).collect()
}

/// Encode PCM i16 samples to A-law bytes.
///
/// Uses the standard ITU-T G.711 A-law algorithm.
pub fn alaw_encode(input: &[i16]) -> Vec<u8> {
    input.iter().map(|&sample| encode_alaw_sample(sample)).collect()
}

fn encode_alaw_sample(sample: i16) -> u8 {
    let mut pcm_val = sample as i32;
    let sign: i32;

    if pcm_val >= 0 {
        sign = 0xD5; // sign bit + toggle bits
    } else {
        sign = 0x55; // toggle bits only
        pcm_val = -pcm_val;
        if pcm_val > 32767 {
            pcm_val = 32767;
        }
    }

    let mask: u8;
    if pcm_val < 256 {
        // Segment 0.
        mask = ((pcm_val >> 4) & 0x0F) as u8;
        // segment = 0, so segment bits are 0
    } else {
        // Find the segment.
        let mut segment: i32 = 1;
        let mut level = 256;
        while segment < 8 {
            level <<= 1;
            if pcm_val < level {
                break;
            }
            segment += 1;
        }
        if segment >= 8 {
            // Clip.
            mask = 0x7F;
            return mask ^ (sign as u8);
        }
        mask = (((segment << 4) | ((pcm_val >> (segment + 3)) & 0x0F)) & 0x7F) as u8;
    }

    mask ^ (sign as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulaw_decode_roundtrip() {
        // Encode then decode should be close to the original for typical speech values.
        let samples: Vec<i16> = vec![0, 100, -100, 1000, -1000, 10000, -10000, 32767, -32768];
        let encoded = ulaw_encode(&samples);
        let decoded = ulaw_decode(&encoded);

        for (orig, dec) in samples.iter().zip(decoded.iter()) {
            // mu-law has ~1% quantization error for large values.
            let diff = (*orig as i32 - *dec as i32).unsigned_abs();
            assert!(
                diff < 1000 || (*orig as i32).unsigned_abs() > 20000,
                "ulaw roundtrip too far off: orig={orig}, decoded={dec}"
            );
        }
    }

    #[test]
    fn ulaw_decode_silence() {
        // mu-law value 0xFF typically maps to 0.
        let decoded = ulaw_decode(&[0xFF]);
        assert_eq!(decoded[0], 0);
    }

    #[test]
    fn ulaw_encode_empty() {
        assert!(ulaw_encode(&[]).is_empty());
    }

    #[test]
    fn ulaw_decode_empty() {
        assert!(ulaw_decode(&[]).is_empty());
    }

    #[test]
    fn alaw_decode_roundtrip() {
        let samples: Vec<i16> = vec![0, 100, -100, 1000, -1000, 10000, -10000, 32767, -32768];
        let encoded = alaw_encode(&samples);
        let decoded = alaw_decode(&encoded);

        for (orig, dec) in samples.iter().zip(decoded.iter()) {
            let diff = (*orig as i32 - *dec as i32).unsigned_abs();
            assert!(
                diff < 1500 || (*orig as i32).unsigned_abs() > 20000,
                "alaw roundtrip too far off: orig={orig}, decoded={dec}"
            );
        }
    }

    #[test]
    fn alaw_encode_empty() {
        assert!(alaw_encode(&[]).is_empty());
    }

    #[test]
    fn alaw_decode_empty() {
        assert!(alaw_decode(&[]).is_empty());
    }

    #[test]
    fn ulaw_decode_table_size() {
        assert_eq!(ULAW_DECODE_TABLE.len(), 256);
    }

    #[test]
    fn alaw_decode_table_size() {
        assert_eq!(ALAW_DECODE_TABLE.len(), 256);
    }

    #[test]
    fn ulaw_encode_symmetry() {
        // Positive and negative of same magnitude should differ only in sign bit.
        let pos = encode_ulaw_sample(1000);
        let neg = encode_ulaw_sample(-1000);
        // The sign bit is bit 7 in the inverted output.
        assert_eq!(pos ^ neg, 0x80);
    }
}
