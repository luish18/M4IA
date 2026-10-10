/* Autoregressive decoding of a tiny language model on the hetero_soc board.
 *
 * The Deeploy-generated graph is one *decode step*: one token in, the logits
 * for the next one out, plus the key and value that token contributes to each
 * attention head. Everything around it is this program, which is the part of
 * an LLM a classifier has no equivalent of:
 *
 *   for each problem:            "07+48=" -> "550;"
 *     clear the KV caches
 *     for t in 0 .. LLM_STEPS-1:
 *       x     = tok_emb[token] + pos_emb[t]        (the embedding lookup)
 *       mask  = 0 for slots 0..t, LLM_MASK_OFF after
 *       sel   = 1 at slot t, 0 before, LLM_SEL_OFF after
 *       RunNetwork()                               (nodes on whichever engine)
 *       write each head's k_new / v_new into slot t of its cache
 *       past the prompt: token = argmax(logits)    (greedy)
 *
 * The prompt goes through the same graph a token at a time -- there is one
 * network per binary, so no separate prefill graph -- which makes every step
 * the M=1 regime: each weight is streamed for a single row of output.
 *
 * Scoring follows runtime/mesh/mnist_main.c. Against llm_answers (the true sum)
 * gives an accuracy that only says whether the model was worth training.
 * Against llm_reference (onnxruntime's greedy tokens, from pipeline/llm.py
 * running this exact loop) is the check that can fail the run: a chip that
 * picks a different token computed something different. Since a wrong token is
 * fed back, one disagreement changes everything after it in that problem; the
 * problem counts as agreeing only if all its answer tokens match.
 *
 * Cycles are kept per position, because what makes decoding interesting is
 * that the work changes as the sequence grows.
 */
#include <stdint.h>
#include <string.h>

#include "Network.h"

#include "bench.h"
#include "hes_host.h"
#include "llm_data.h"
#include "miniio.h"

/* Cap the run, so a quick check does not have to do all of them. */
#ifndef HES_SAMPLES
#define HES_SAMPLES LLM_NUM_PROBLEMS
#endif

#if HES_SAMPLES < LLM_NUM_PROBLEMS
#define RUN_PROBLEMS HES_SAMPLES
#else
#define RUN_PROBLEMS LLM_NUM_PROBLEMS
#endif

#define NB_IN (3 + 2 * LLM_N_LAYER * LLM_N_HEAD)
#define NB_OUT (1 + 2 * LLM_N_LAYER * LLM_N_HEAD)

/* The KV caches, owned by the host. K is stored transposed -- [d_head][ctx],
 * one column per position -- because that is the operand the graph's score
 * MatMul wants; V is [ctx][d_head], one row per position. */
static float kt_cache[LLM_N_LAYER][LLM_N_HEAD][LLM_D_HEAD][LLM_CTX];
static float v_cache[LLM_N_LAYER][LLM_N_HEAD][LLM_CTX][LLM_D_HEAD];

/* Network cycles summed over problems, per position. */
static uint64_t pos_cycles[LLM_STEPS];

static uint32_t argmax(const float *p, uint32_t n) {
  uint32_t best = 0;
  for (uint32_t i = 1; i < n; i++) {
    if (p[i] > p[best]) {
      best = i;
    }
  }
  return best;
}

/* The generated network has to have exactly the interface pipeline/llm.py
 * exported, in the same order; a graph that came out different would
 * otherwise be fed the wrong buffers without complaint. */
static int check_interface(void) {
  int ok = DeeployNetwork_num_inputs == NB_IN && DeeployNetwork_num_outputs == NB_OUT;
  if (ok) {
    ok = DeeployNetwork_inputs_bytes[LLM_IN_X] == LLM_D_MODEL * sizeof(float) &&
         DeeployNetwork_inputs_bytes[LLM_IN_MASK] == LLM_CTX * sizeof(float) &&
         DeeployNetwork_inputs_bytes[LLM_IN_SEL] == LLM_CTX * sizeof(float) &&
         DeeployNetwork_outputs_bytes[LLM_OUT_LOGITS] == LLM_VOCAB * sizeof(float);
    for (uint32_t l = 0; ok && l < LLM_N_LAYER; l++) {
      for (uint32_t h = 0; ok && h < LLM_N_HEAD; h++) {
        ok = DeeployNetwork_inputs_bytes[LLM_IN_KT(l, h)] == sizeof(kt_cache[l][h]) &&
             DeeployNetwork_inputs_bytes[LLM_IN_V(l, h)] == sizeof(v_cache[l][h]) &&
             DeeployNetwork_outputs_bytes[LLM_OUT_K(l, h)] ==
                 LLM_D_HEAD * sizeof(float) &&
             DeeployNetwork_outputs_bytes[LLM_OUT_V(l, h)] == LLM_D_HEAD * sizeof(float);
      }
    }
  }
  if (!ok) {
    print_str("[HES-ERR] the generated network's inputs/outputs do not match "
              "llm_data.h\n");
  }
  return ok;
}

/* Fill the graph inputs for the token at position t. */
static void stage_step(uint32_t token, uint32_t t) {
  float *x = (float *)DeeployNetwork_inputs[LLM_IN_X];
  for (uint32_t i = 0; i < LLM_D_MODEL; i++) {
    x[i] = llm_tok_emb[token][i] + llm_pos_emb[t][i];
  }

  float *mask = (float *)DeeployNetwork_inputs[LLM_IN_MASK];
  float *sel = (float *)DeeployNetwork_inputs[LLM_IN_SEL];
  for (uint32_t s = 0; s < LLM_CTX; s++) {
    mask[s] = s <= t ? 0.0f : LLM_MASK_OFF;
    sel[s] = s < t ? 0.0f : (s == t ? 1.0f : LLM_SEL_OFF);
  }

  for (uint32_t l = 0; l < LLM_N_LAYER; l++) {
    for (uint32_t h = 0; h < LLM_N_HEAD; h++) {
      memcpy(DeeployNetwork_inputs[LLM_IN_KT(l, h)], kt_cache[l][h],
             sizeof(kt_cache[l][h]));
      memcpy(DeeployNetwork_inputs[LLM_IN_V(l, h)], v_cache[l][h],
             sizeof(v_cache[l][h]));
    }
  }
}

/* Append the step's keys and values to the caches, at slot t. */
static void absorb_step(uint32_t t) {
  for (uint32_t l = 0; l < LLM_N_LAYER; l++) {
    for (uint32_t h = 0; h < LLM_N_HEAD; h++) {
      const float *k = (const float *)DeeployNetwork_outputs[LLM_OUT_K(l, h)];
      const float *v = (const float *)DeeployNetwork_outputs[LLM_OUT_V(l, h)];
      for (uint32_t d = 0; d < LLM_D_HEAD; d++) {
        kt_cache[l][h][d][t] = k[d];
        v_cache[l][h][t][d] = v[d];
      }
    }
  }
}

int main(void) {
  InitNetwork(0, 1);
  hes_engine_init();

  for (uint32_t e = HES_ENGINE_SNITCH; e < HES_NB_ENGINES; e++) {
    if (!hes_engine_ready(e)) {
      print_str("[HES-ERR] cluster ");
      print_str(hes_engine_name(e));
      print_str(" never signed in\n");
    }
  }
  if (!check_interface()) {
    return 1;
  }

  uint32_t correct = 0, agree = 0;
  uint64_t net_cycles = 0;
  const uint32_t total_steps = RUN_PROBLEMS * LLM_STEPS;

  uint64_t c0 = read_mcycle();

  for (uint32_t n = 0; n < RUN_PROBLEMS; n++) {
    memset(kt_cache, 0, sizeof(kt_cache));
    memset(v_cache, 0, sizeof(v_cache));

    uint8_t seq[LLM_PROMPT_LEN + LLM_ANSWER_LEN];
    memcpy(seq, llm_prompts[n], LLM_PROMPT_LEN);

    for (uint32_t t = 0; t < LLM_STEPS; t++) {
      hes_set_sample(n * LLM_STEPS + t + 1, total_steps);
      stage_step(seq[t], t);

      uint64_t s0 = read_mcycle();
      RunNetwork(0, 1);
      uint64_t s1 = read_mcycle();
      pos_cycles[t] += s1 - s0;
      net_cycles += s1 - s0;

      absorb_step(t);
      if (t + 1 >= LLM_PROMPT_LEN) {
        seq[t + 1] = (uint8_t)argmax(
            (const float *)DeeployNetwork_outputs[LLM_OUT_LOGITS], LLM_VOCAB);
      }
    }

    const uint8_t *answer = &seq[LLM_PROMPT_LEN];
    if (memcmp(answer, llm_answers[n], LLM_ANSWER_LEN) == 0) {
      correct++;
    }
    if (memcmp(answer, llm_reference[n], LLM_ANSWER_LEN) == 0) {
      agree++;
    } else {
      print_str("[HES-ERR] problem ");
      print_u64(n);
      print_str(": chip answers");
      for (uint32_t i = 0; i < LLM_ANSWER_LEN; i++) {
        print_str(" ");
        print_u64(answer[i]);
      }
      print_str(", onnxruntime");
      for (uint32_t i = 0; i < LLM_ANSWER_LEN; i++) {
        print_str(" ");
        print_u64(llm_reference[n][i]);
      }
      print_str("\n");
    }
  }

  uint64_t total = read_mcycle() - c0;

  /* Prompt steps consume a known token; answer steps consume one the model
   * generated. The graph does the same work either way -- the split is there
   * because a separate prefill graph would change only the first part. */
  uint64_t prompt_cycles = 0, answer_cycles = 0;
  for (uint32_t t = 0; t < LLM_STEPS; t++) {
    if (t < LLM_PROMPT_LEN) {
      prompt_cycles += pos_cycles[t];
    } else {
      answer_cycles += pos_cycles[t];
    }
    print_str("[HES-LLM-POS] pos=");
    print_u64(t);
    print_str(" cycles=");
    print_u64(pos_cycles[t] / (RUN_PROBLEMS ? RUN_PROBLEMS : 1));
    print_str("\n");
  }

  print_str("[HES-LLM] problems=");
  print_u64(RUN_PROBLEMS);
  print_str(" correct=");
  print_u64(correct);
  print_str(" agree_with_onnx=");
  print_u64(agree);
  print_str(" steps=");
  print_u64(total_steps);
  print_str(" cycles_total=");
  print_u64(total);
  print_str(" cycles_per_step=");
  print_u64(net_cycles / (total_steps ? total_steps : 1));
  print_str(" prompt_cycles=");
  print_u64(prompt_cycles);
  print_str(" answer_cycles=");
  print_u64(answer_cycles);
  /* Embedding, mask/sel, cache copies and argmax: the host's own share of
   * decoding, outside any graph node. */
  print_str(" host_cycles=");
  print_u64(total - net_cycles);
  print_str(" offload_failures=");
  print_u64(hes_failures());
  print_str("\n");

  /* The run is a failure if the chip disagreed with the reference model or an
   * offload failed -- not if the model got a sum wrong. */
  return (agree == RUN_PROBLEMS && hes_failures() == 0) ? 0 : 1;
}
