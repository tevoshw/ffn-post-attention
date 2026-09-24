# Post-Attention

After attention, each token has absorbed information from all other tokens via attention scores. Shape: (B, T, D_MODEL).

**Problem:** Stacking layers causes gradient vanishing. Deep networks lose signal.

**Solution:**

1. **Residual Connection**: x = x + attention_output
   - Allows gradient to flow directly through layers
   - Without it: backprop dies in 12+ layers
   - Information from attention + original token preserved

2. **Layer Normalization**: x = LayerNorm(x)
   - Attention output can have extreme values (large scores)
   - LayerNorm keeps mean=0, variance=1 per token
   - Prevents exploding values from accumulating
   - Stabilizes training

```python
x = x + attention(x)  # Add original + attention
x = LayerNorm(x)      # Normalize range
```

# Feed Forward Network

Each token now goes through dense layers **independently** (no token-to-token interaction).

**Why dense layer?**
- Attention: communication between tokens
- FFN: reasoning/processing within each token
- Non-linearity needed (pure linear = useless)

**Architecture:**

```python
# Expand space to capture more patterns
expanded = linear_1(x)  # D_MODEL → 4×D_MODEL

# Non-linearity (GELU better than ReLU)
activated = gelu(expanded)

# Contract back to original dimension
output = linear_2(activated)  # 4×D_MODEL → D_MODEL
```

**Why 4×?**
- Emperical from paper
- 4× gives good capacity/speed tradeoff
- Smaller = underfitting, Larger = overkill

**Then repeat residual + norm:**

```python
x = x + output       # Residual (same reason as attention)
x = LayerNorm(x)     # Normalize (same reason)
```

**Why 66% of params here?**
- Linear layers have most params: (D × 4D) + (4D × D)
- Attention has: Q, K, V, Output lineares = smaller
- FFN layers × 12 = massive
