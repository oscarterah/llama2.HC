// Inference for Llama-2 Transformer model in HolyC
// Ported from run.c

class Config {
  I32 dim;
  I32 hidden_dim;
  I32 n_layers;
  I32 n_heads;
  I32 n_kv_heads;
  I32 vocab_size;
  I32 seq_len;
};

class TransformerWeights {
  F64 *token_embedding_table;
  F64 *rms_att_weight;
  F64 *rms_ffn_weight;
  F64 *wq;
  F64 *wk;
  F64 *wv;
  F64 *wo;
  F64 *w1;
  F64 *w2;
  F64 *w3;
  F64 *rms_final_weight;
  F64 *wcls;
};

class RunState {
  F64 *x;
  F64 *xb;
  F64 *xb2;
  F64 *hb;
  F64 *hb2;
  F64 *q;
  F64 *k;
  F64 *v;
  F64 *att;
  F64 *logits;
  F64 *key_cache;
  F64 *value_cache;
};

class Transformer {
  Config config;
  TransformerWeights weights;
  RunState state;
  F64 *w_data; // weights converted to F64
};

class TokenIndex {
  U8 *str;
  I64 id;
};

class Tokenizer {
  U8 **vocab;
  F64 *vocab_scores;
  TokenIndex *sorted_vocab;
  I64 vocab_size;
  U32 max_token_length;
  U8 byte_pieces[512];
};

class ProbIndex {
  F64 prob;
  I64 index;
};

class Sampler {
  I64 vocab_size;
  ProbIndex *probindex;
  F64 temperature;
  F64 topp;
  U64 rng_state;
};

F64 F32ToF64(U32 u) {
  U64 sign = u & 0x80000000;
  sign <<= 32;
  I64 exp = (u >> 23) & 0xFF;
  U64 mant = u & 0x7FFFFF;
  if (exp == 0) {
    if (mant == 0) return *(&sign)(F64 *);
    return 0.0;
  }
  if (exp == 255) return 0.0;
  exp = exp - 127 + 1023;
  U64 res = sign | (exp << 52) | (mant << 29);
  return *(&res)(F64 *);
}

U0 MallocRunState(RunState *s, Config *p) {
  I64 kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  s->x = CAlloc(p->dim * sizeof(F64));
  s->xb = CAlloc(p->dim * sizeof(F64));
  s->xb2 = CAlloc(p->dim * sizeof(F64));
  s->hb = CAlloc(p->hidden_dim * sizeof(F64));
  s->hb2 = CAlloc(p->hidden_dim * sizeof(F64));
  s->q = CAlloc(p->dim * sizeof(F64));
  s->key_cache = CAlloc(p->n_layers * p->seq_len * kv_dim * sizeof(F64));
  s->value_cache = CAlloc(p->n_layers * p->seq_len * kv_dim * sizeof(F64));
  s->att = CAlloc(p->n_heads * p->seq_len * sizeof(F64));
  s->logits = CAlloc(p->vocab_size * sizeof(F64));
}

U0 FreeRunState(RunState *s) {
  Free(s->x); Free(s->xb); Free(s->xb2);
  Free(s->hb); Free(s->hb2); Free(s->q);
  Free(s->att); Free(s->logits);
  Free(s->key_cache); Free(s->value_cache);
}

U0 MapWeights(TransformerWeights *w, Config *p, F64 *ptr, I64 shared_weights) {
  I64 head_size = p->dim / p->n_heads;
  I64 n_layers = p->n_layers;
  w->token_embedding_table = ptr;
  ptr += p->vocab_size * p->dim;
  w->rms_att_weight = ptr;
  ptr += n_layers * p->dim;
  w->wq = ptr;
  ptr += n_layers * p->dim * (p->n_heads * head_size);
  w->wk = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wv = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wo = ptr;
  ptr += n_layers * (p->n_heads * head_size) * p->dim;
  w->rms_ffn_weight = ptr;
  ptr += n_layers * p->dim;
  w->w1 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->w2 = ptr;
  ptr += n_layers * p->hidden_dim * p->dim;
  w->w3 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->rms_final_weight = ptr;
  ptr += p->dim;
  ptr += p->seq_len * head_size; // skip RoPE freqs
  if (shared_weights) w->wcls = w->token_embedding_table;
  else w->wcls = ptr;
}

U0 BuildTransformer(Transformer *t, U8 *checkpoint_path) {
  I64 i, size, num_weights, shared_weights;
  U8 *buf, *f32_ptr;
  Config *c_ptr;
  I64 *sz_ptr = &size;

  buf = FileRead(checkpoint_path, sz_ptr);
  if (!buf) return;
  c_ptr = &t->config;
  MemCpy(c_ptr, buf, sizeof(Config));
  shared_weights = t->config.vocab_size > 0;
  t->config.vocab_size = Abs(t->config.vocab_size);

  num_weights = (size - sizeof(Config)) / 4;
  t->w_data = MAlloc(num_weights * sizeof(F64));
  f32_ptr = buf + sizeof(Config);
  for (i = 0; i < num_weights; i++) {
    t->w_data[i] = F32ToF64(f32_ptr(U32*)[i]);
  }
  Free(buf);

  MapWeights(&t->weights, c_ptr, t->w_data, shared_weights);
  MallocRunState(&t->state, c_ptr);
}

U0 FreeTransformer(Transformer *t) {
  Free(t->w_data);
  FreeRunState(&t->state);
}

U0 RMSNorm(F64 *o, F64 *x, F64 *weight, I64 size) {
  I64 j;
  F64 ss = 0.0;
  for (j = 0; j < size; j++) ss += x[j] * x[j];
  ss /= size;
  ss += 1e-5;
  ss = 1.0 / Sqrt(ss);
  for (j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

U0 Softmax(F64 *x, I64 size) {
  I64 i;
  F64 max_val = x[0], sum = 0.0;
  for (i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
  for (i = 0; i < size; i++) {
    x[i] = Exp(x[i] - max_val);
    sum += x[i];
  }
  for (i = 0; i < size; i++) x[i] /= sum;
}

U0 MatMul(F64 *xout, F64 *x, F64 *w, I64 n, I64 d) {
  I64 i, j;
  for (i = 0; i < d; i++) {
    F64 val = 0.0;
    for (j = 0; j < n; j++) val += w[i * n + j] * x[j];
    xout[i] = val;
  }
}

F64* Forward(Transformer *transformer, I64 token, I64 pos) {
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;
  F64 *x = s->x;
  I64 dim = p->dim;
  I64 kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  I64 kv_mul = p->n_heads / p->n_kv_heads;
  I64 hidden_dim = p->hidden_dim;
  I64 head_size = dim / p->n_heads;
  I64 l, i, h, t, loff, rotn, head_dim;
  F64 *content_row, *vec, *q, *att, *k, *v, *xb;
  F64 freq, val, fcr, fci, v0, v1, score, a;

  content_row = w->token_embedding_table + token * dim;
  MemCpy(x, content_row, dim * sizeof(F64));

  for (l = 0; l < p->n_layers; l++) {
    RMSNorm(s->xb, x, w->rms_att_weight + l * dim, dim);
    loff = l * p->seq_len * kv_dim;
    s->k = s->key_cache + loff + pos * kv_dim;
    s->v = s->value_cache + loff + pos * kv_dim;

    MatMul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
    MatMul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
    MatMul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

    for (i = 0; i < dim; i += 2) {
      head_dim = i % head_size;
      freq = 1.0 / Pow(10000.0, head_dim(F64) / head_size);
      val = pos * freq;
      fcr = Cos(val); fci = Sin(val);
      if (i < kv_dim) rotn = 2; else rotn = 1;
      for (h = 0; h < rotn; h++) {
        if (h == 0) vec = s->q; else vec = s->k;
        v0 = vec[i]; v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
      }
    }

    for (h = 0; h < p->n_heads; h++) {
      q = s->q + h * head_size;
      att = s->att + h * p->seq_len;
      for (t = 0; t <= pos; t++) {
        k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        score = 0.0;
        for (i = 0; i < head_size; i++) score += q[i] * k[i];
        att[t] = score / Sqrt(head_size);
      }
      Softmax(att, pos + 1);
      xb = s->xb + h * head_size;
      MemSet(xb, 0, head_size * sizeof(F64));
      for (t = 0; t <= pos; t++) {
        v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        a = att[t];
        for (i = 0; i < head_size; i++) xb[i] += a * v[i];
      }
    }

    MatMul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
    for (i = 0; i < dim; i++) x[i] += s->xb2[i];
    RMSNorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);
    MatMul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
    MatMul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
    for (i = 0; i < hidden_dim; i++) {
      val = s->hb[i];
      val *= (1.0 / (1.0 + Exp(-val)));
      val *= s->hb2[i];
      s->hb[i] = val;
    }
    MatMul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);
    for (i = 0; i < dim; i++) x[i] += s->xb[i];
  }
  RMSNorm(x, x, w->rms_final_weight, dim);
  MatMul(s->logits, x, w->wcls, p->dim, p->vocab_size);
  return s->logits;
}

I64 CompareTokens(U8 *e1, U8 *e2) {
  TokenIndex *a = e1(TokenIndex*), *b = e2(TokenIndex*);
  return StrCmp(a->str, b->str);
}

U0 BuildTokenizer(Tokenizer *t, U8 *tokenizer_path, I64 vocab_size) {
  I64 i, size, len;
  U8 *buf, *ptr;
  I64 *sz_ptr = &size;
  t->vocab_size = vocab_size;
  t->vocab = MAlloc(vocab_size * sizeof(U8*));
  t->vocab_scores = MAlloc(vocab_size * sizeof(F64));
  t->sorted_vocab = NULL;
  for (i = 0; i < 256; i++) {
    t->byte_pieces[i * 2] = i;
    t->byte_pieces[i * 2 + 1] = 0;
  }
  buf = FileRead(tokenizer_path, sz_ptr);
  if (!buf) return;
  ptr = buf;
  t->max_token_length = *ptr(U32*); ptr += 4;
  for (i = 0; i < vocab_size; i++) {
    t->vocab_scores[i] = F32ToF64(*ptr(U32*)); ptr += 4;
    len = *ptr(I32*); ptr += 4;
    t->vocab[i] = MAlloc(len + 1);
    MemCpy(t->vocab[i], ptr, len);
    t->vocab[i][len] = 0;
    ptr += len;
  }
  Free(buf);
}

U0 FreeTokenizer(Tokenizer *t) {
  I64 i;
  for (i = 0; i < t->vocab_size; i++) Free(t->vocab[i]);
  Free(t->vocab); Free(t->vocab_scores); Free(t->sorted_vocab);
}

U8* Decode(Tokenizer *t, I64 prev_token, I64 token) {
  U8 *piece = t->vocab[token];
  if (prev_token == 1 && piece[0] == ' ') piece++;
  return piece;
}

U0 SafePrint(U8 *piece) {
  if (!piece || !piece[0]) return;
  Print("%s", piece);
}

// Fixed BSearch with correct HolyC syntax
U8* BSearch(U8 *key, U8 *base, I64 num, I64 width, I64 (*fp_compare)(U8 *e1, U8 *e2)) {
  I64 low = 0, high = num - 1, mid, res;
  while (low <= high) {
    mid = (low + high) / 2;
    res = fp_compare(key, base + mid * width);
    if (res < 0) high = mid - 1;
    else if (res > 0) low = mid + 1;
    else return base + mid * width;
  }
  return NULL;
}

U0 QuickSort(U8 *base, I64 num, I64 width, I64 (*fp_compare)(U8 *e1, U8 *e2)) {
  if (num < 2) return;
  U8 *pivot = base + (num / 2) * width, *tmp = MAlloc(width);
  I64 i = 0, j = num - 1;
  while (i <= j) {
    while (fp_compare(base + i * width, pivot) < 0) i++;
    while (fp_compare(base + j * width, pivot) > 0) j--;
    if (i <= j) {
      MemCpy(tmp, base + i * width, width);
      MemCpy(base + i * width, base + j * width, width);
      MemCpy(base + j * width, tmp, width);
      i++; j--;
    }
  }
  Free(tmp);
  if (j + 1 > 1) QuickSort(base, j + 1, width, fp_compare);
  if (num - i > 1) QuickSort(base + i * width, num - i, width, fp_compare);
}

I64 StrLookup(U8 *str, TokenIndex *sorted_vocab, I64 vocab_size) {
  TokenIndex tok, *tok_ptr, *result;
  tok.str = str;
  tok_ptr = &tok;
  result = BSearch(tok_ptr(U8*), sorted_vocab(U8*), vocab_size, sizeof(TokenIndex), &CompareTokens)(TokenIndex*);
  if (result) return result->id;
  return -1;
}

U0 Encode(Tokenizer *t, U8 *text, I64 bos, I64 eos, I64 *tokens, I64 *n_tokens) {
  I64 i, id;
  if (!text) return;
  if (!t->sorted_vocab) {
    t->sorted_vocab = MAlloc(t->vocab_size * sizeof(TokenIndex));
    for (i = 0; i < t->vocab_size; i++) {
      t->sorted_vocab[i].str = t->vocab[i];
      t->sorted_vocab[i].id = i;
    }
    QuickSort(t->sorted_vocab(U8*), t->vocab_size, sizeof(TokenIndex), &CompareTokens);
  }
  *n_tokens = 0;
  if (bos) tokens[(*n_tokens)++] = 1;
  if (text[0]) {
    id = StrLookup(" ", t->sorted_vocab, t->vocab_size);
    if (id != -1) tokens[(*n_tokens)++] = id;
  }
  // Simplified encoding for demonstration
  U8 *c = text;
  while (*c) {
    U8 buf[2]; buf[0] = *c; buf[1] = 0;
    id = StrLookup(buf, t->sorted_vocab, t->vocab_size);
    if (id != -1) tokens[(*n_tokens)++] = id;
    c++;
  }
  if (eos) tokens[(*n_tokens)++] = 2;
}

I64 SampleArgmax(F64 *probabilities, I64 n) {
  I64 i, max_i = 0;
  F64 max_p = probabilities[0];
  for (i = 1; i < n; i++) if (probabilities[i] > max_p) { max_i = i; max_p = probabilities[i]; }
  return max_i;
}

U32 RandomU32(U64 *state) {
  *state ^= *state >> 12; *state ^= *state << 25; *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1D) >> 32;
}

F64 RandomF32(U64 *state) { return (RandomU32(state) >> 8) / 16777216.0; }

I64 SampleMult(F64 *probabilities, I64 n, F64 coin) {
  I64 i;
  F64 cdf = 0.0;
  for (i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) return i;
  }
  return n - 1;
}

I64 Sample(Sampler *sampler, F64 *logits) {
  I64 next, q;
  F64 coin;
  if (sampler->temperature == 0.0) next = SampleArgmax(logits, sampler->vocab_size);
  else {
    for (q = 0; q < sampler->vocab_size; q++) logits[q] /= sampler->temperature;
    Softmax(logits, sampler->vocab_size);
    coin = RandomF32(&sampler->rng_state);
    next = SampleMult(logits, sampler->vocab_size, coin);
  }
  return next;
}

U0 Generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, U8 *prompt, I64 steps) {
  I64 num_prompt_tokens = 0, next, token, pos;
  I64 *prompt_tokens;
  F64 start = 0, end;
  F64 *logits;
  U8 *piece;

  if (!prompt) prompt = "";
  prompt_tokens = MAlloc((StrLen(prompt) + 3) * sizeof(I64));
  Encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

  token = prompt_tokens[0];
  pos = 0;
  while (pos < steps) {
    logits = Forward(transformer, token, pos);
    if (pos < num_prompt_tokens - 1) next = prompt_tokens[pos + 1];
    else next = Sample(sampler, logits);
    pos++;
    if (next == 1) break;
    piece = Decode(tokenizer, token, next);
    SafePrint(piece);
    token = next;
    if (start == 0) start = tS;
  }
  Print("\n");
  if (pos > 1) { end = tS; Print("achieved tok/s: %f\n", (pos - 1) / (end - start)); }
  Free(prompt_tokens);
}

U0 Llama2Run(U8 *checkpoint_path, U8 *tokenizer_path="tokenizer.bin", F64 temperature=1.0, F64 topp=0.9, I64 steps=256, U8 *prompt=NULL, U64 rng_seed=0) {
  Transformer transformer; Tokenizer tokenizer; Sampler sampler;
  if (rng_seed == 0) rng_seed = Now;
  BuildTransformer(&transformer, checkpoint_path);
  if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;
  BuildTokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
  sampler.vocab_size = transformer.config.vocab_size;
  sampler.temperature = temperature;
  sampler.topp = topp;
  sampler.rng_state = rng_seed;
  sampler.probindex = MAlloc(sampler.vocab_size * sizeof(ProbIndex));
  Generate(&transformer, &tokenizer, &sampler, prompt, steps);
  Free(sampler.probindex);
  FreeTokenizer(&tokenizer);
  FreeTransformer(&transformer);
}
