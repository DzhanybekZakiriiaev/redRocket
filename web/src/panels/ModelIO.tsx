/* ────────────────────────────────────────────────────────────────────────
   ModelIO.tsx — what the model was actually asked, and what it actually said.

   Two halves, kept visually distinct on purpose:

     RECORDED  — posterior, action, ground truth and rationale. These are in
                 the trace today; every arm emits them.
     EXACT I/O — the literal prompt string and the raw reply. These exist only
                 in a trace enriched by tools/replay_evidence.py.

   When the second half is missing this panel says so and names the command
   that produces it. It does NOT reconstruct an approximation of the prompt
   from the trace: a plausible-looking prompt that was never actually sent is
   worse than an empty panel, because it invites a claim we cannot support.
   ──────────────────────────────────────────────────────────────────────── */

import { useEffect } from 'react'
import { useStore } from '../store'
import { clock } from '../clock'
import { CAUSE_WORD, ACTION_WORD, tPlus, type Cause } from '../trace'

const mono = '"Roboto Mono", ui-monospace, SFMono-Regular, Menlo, monospace'

function Section({ title, note, children }: {
  title: string; note?: string; children: React.ReactNode
}) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 9, marginBottom: 6 }}>
        <span className="t-micro" style={{ color: 'var(--accent)' }}>{title}</span>
        {note && <span className="t-micro faint" style={{ fontSize: 8 }}>{note}</span>}
      </div>
      {children}
    </div>
  )
}

export default function ModelIO() {
  const open = useStore(s => s.modelIO)
  const setModelIO = useStore(s => s.setModelIO)
  const trace = useStore(s => s.trace)
  const selected = useStore(s => s.selected)

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setModelIO(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, setModelIO])

  if (!open || !trace) return null

  const fi = Math.max(0, Math.min(trace.nFrames - 1, Math.round(clock.frame)))
  const series = trace.samples[selected]
  const current = series?.[fi]

  /* Only ~390 of 2304 decisions involved an adverse link, so the model was
     invoked on 17% of frames. Opening this panel at an arbitrary moment
     therefore almost always landed on "no call made", which is accurate and
     useless. Fall back to the nearest decision that DID reason — searching
     backwards first, since the most recent diagnosis is the relevant one —
     and label it plainly when it is not the current frame. Everything below
     then describes that one decision, so reasoning, posterior and result can
     never come from different moments. */
  let shownFrame = -1
  if (current?.ev) {
    shownFrame = fi
  } else if (series) {
    for (let k = fi; k >= 0; k--) if (series[k]?.ev) { shownFrame = k; break }
    if (shownFrame < 0)
      for (let k = fi; k < trace.nFrames; k++) if (series[k]?.ev) { shownFrame = k; break }
  }
  const sample = shownFrame >= 0 ? series[shownFrame] : current
  const offCurrent = shownFrame >= 0 && shownFrame !== fi
  const epoch = trace.meta.epoch_ms
  const advisor = (trace.meta.advisor || '').toUpperCase()
  const causes = trace.meta.causes as Cause[]
  const bars = sample
    ? causes.map((c, i) => ({ c, p: sample.d[i] ?? 0 })).sort((a, b) => b.p - a.p)
    : []
  const correct = sample && sample.truth !== 'nominal' ? sample.cause === sample.truth : null
  const ev = sample?.ev

  return (
    <div
      onClick={() => setModelIO(false)}
      style={{
        position: 'fixed', inset: 0, zIndex: 200,
        background: 'rgba(0,0,0,0.72)',
        display: 'grid', placeItems: 'center', padding: 24,
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        className="panel"
        style={{
          width: 'min(760px, 94vw)', maxHeight: '86vh', overflowY: 'auto',
          padding: '16px 18px', background: 'var(--bg, #07090b)',
          border: '1px solid var(--ink-dim)',
        }}
      >
        {/* ── header ── */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
          <div>
            <span className="t-title acc">{selected}</span>
            <span className="t-sub" style={{ marginLeft: 10 }}>MODEL INPUT / OUTPUT</span>
          </div>
          <button
            onClick={() => setModelIO(false)}
            className="t-micro"
            style={{
              cursor: 'pointer', background: 'transparent',
              border: '1px solid var(--ink-faint)', padding: '2px 8px',
            }}
          >
            ESC ✕
          </button>
        </div>
        <div className="t-micro dim" style={{ marginTop: 4, marginBottom: offCurrent ? 8 : 16 }}>
          {advisor} · DECISION AT T+{sample ? tPlus(sample.sampledAt, epoch) : '—'}
          {ev?.peer ? ` · LINK ${selected} ↔ ${ev.peer}` : ''}
        </div>
        {offCurrent && (
          <div
            className="t-micro"
            style={{
              marginBottom: 16, padding: '6px 9px',
              border: '1px solid var(--ink-faint)', color: 'var(--neutral)',
              lineHeight: 1.5,
            }}
          >
            {selected} IS NOMINAL RIGHT NOW — SHOWING ITS MOST RECENT ACTUAL
            DIAGNOSIS. THE MODEL IS ONLY INVOKED WHEN A LINK IS ADVERSE.
          </div>
        )}

        {!sample ? (
          <div className="t-body dim">NO BELIEF SAMPLE AT THIS FRAME.</div>
        ) : (
          <>
            {/* ── EXACT I/O — only from an enriched trace ── */}
            {ev ? (
              <>
                {ev.steps && ev.steps.length > 0 && (
                  <Section title="MODEL REASONING" note="SAMPLED BEFORE THE ANSWER — NOT WRITTEN AFTER IT">
                    <div style={{ border: '1px solid var(--ink-faint)', padding: '10px 12px' }}>
                      {ev.steps.map((s, i) => (
                        <div key={i} style={{ display: 'flex', gap: 9, marginBottom: 6 }}>
                          <span className="t-micro" style={{ color: 'var(--accent)', flex: '0 0 auto', width: 14 }}>
                            {String(i + 1).padStart(2, '0')}
                          </span>
                          <span className="t-body" style={{ lineHeight: 1.45, textTransform: 'none' }}>{s}</span>
                        </div>
                      ))}
                      <div className="t-micro faint" style={{ marginTop: 9, fontSize: 8, lineHeight: 1.5 }}>
                        THESE TOKENS WERE GENERATED AHEAD OF THE CAUSE, SO THEY WERE IN
                        CONTEXT WHEN IT WAS CHOSEN.
                      </div>
                    </div>
                  </Section>
                )}

                {/* The verbatim prompt and raw reply are deliberately NOT shown.
                    They are still carried in the trace (ev.prompt / ev.raw) for
                    anyone who wants to audit them, but on screen a 1700-character
                    wall of cause-menu text buried the one thing worth reading.
                    The evidence table below is the same information, legibly. */}

                {ev.f && Object.keys(ev.f).length > 0 && (
                  <Section title="EVIDENCE FIELDS" note="AS EXTRACTED FROM THE OBSERVATION">
                    {Object.entries(ev.f).map(([k, v]) => (
                      <div key={k} className="t-micro" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                        <span className="dim" style={{ textTransform: 'none' }}>{k}</span>
                        <span>{String(v)}</span>
                      </div>
                    ))}
                  </Section>
                )}
              </>
            ) : trace.hasEvidence ? (
              /* The trace DOES carry model I/O — this decision just never
                 produced a prompt. That is not a gap, it is the design: with
                 no adverse link there is nothing to diagnose, so gemma.py
                 returns nominal without spending a call. Worth saying out
                 loud, because "the model is only invoked when something is
                 actually wrong" is a cost argument in its own right. */
              <Section title="PROMPT SENT TO MODEL" note="NO CALL MADE AT THIS DECISION">
                <div
                  style={{
                    border: '1px solid var(--ink-faint)', padding: '11px 13px',
                    background: 'rgba(var(--ink-rgb),0.04)',
                  }}
                >
                  <div className="t-body" style={{ color: 'var(--neutral)', marginBottom: 7 }}>
                    NO LINK WAS ADVERSE — THE MODEL WAS NOT INVOKED.
                  </div>
                  <div className="t-micro dim" style={{ lineHeight: 1.6, textTransform: 'none', letterSpacing: 0 }}>
                    Every link on this asset last reported OK, so there was nothing to
                    diagnose. <span style={{ color: 'var(--ink)' }}>gemma.py</span> returns a
                    nominal belief and spends no inference — the model runs only when
                    something is actually wrong.
                    <br /><br />
                    To see a real prompt, open{' '}
                    <span style={{ color: 'var(--accent)' }}>⚑ RUN SCENARIO</span> and pick a
                    fault, or scrub to a window where this asset is not nominal.
                  </div>
                </div>
              </Section>
            ) : (
              <Section title="PROMPT SENT TO MODEL" note="NOT PRESENT IN THIS TRACE">
                <div
                  style={{
                    border: '1px solid var(--alert)', padding: '11px 13px',
                    background: 'rgba(var(--ink-rgb),0.04)',
                  }}
                >
                  <div className="t-body" style={{ color: 'var(--alert)', marginBottom: 7 }}>
                    THE MODEL'S INPUT WAS NEVER EXPORTED.
                  </div>
                  <div className="t-micro dim" style={{ lineHeight: 1.6, textTransform: 'none', letterSpacing: 0 }}>
                    sim/trace.py emits the posterior, action, ground truth and rationale —
                    but not the prompt that produced them. The evidence bullets are built
                    in <span style={{ color: 'var(--ink)' }}>sim/advisor/gemma.py:_prompt</span> and
                    discarded after the call.
                    <br /><br />
                    To fill this panel with the verbatim prompt, regenerate the trace with:
                    <div style={{
                      font: `10px ${mono}`, color: 'var(--accent)', marginTop: 7,
                      padding: '7px 9px', border: '1px solid var(--ink-faint)',
                    }}>python -m tools.replay_evidence</div>
                    <div style={{ marginTop: 7 }}>
                      That replays the recorded decisions through the real prompt builder,
                      so it needs no API key and cannot change the run.
                    </div>
                  </div>
                </div>
              </Section>
            )}

            <div style={{ height: 1, background: 'var(--ink-faint)', margin: '18px 0 16px' }} />

            {/* ── RECORDED — always available ── */}
            <Section title="POSTERIOR" note="IN TRACE · SUMS TO 1">
              {bars.map(({ c, p }, i) => (
                <div key={c} style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 4 }}>
                  <span className={i === 0 ? '' : 'dim'} style={{ width: 88, fontSize: 9, letterSpacing: '0.1em' }}>
                    {CAUSE_WORD[c]}
                  </span>
                  <div style={{ flex: 1, height: 6, background: 'rgba(var(--ink-rgb),0.08)' }}>
                    <div style={{
                      width: `${Math.max(1, p * 100)}%`, height: '100%',
                      background: i === 0 ? 'var(--accent)' : 'rgba(var(--ink-rgb),0.35)',
                    }} />
                  </div>
                  <span className="dim" style={{ width: 30, textAlign: 'right', fontSize: 9 }}>
                    {(p * 100).toFixed(1)}
                  </span>
                </div>
              ))}
            </Section>

            {/* On the baseline arm the rationale key comes AFTER cause in the
                template, so it is a justification for a choice already made.
                Say so, rather than letting it borrow the authority of the
                reasoning-first arm's steps. */}
            {sample.rationale && (
              <Section
                title={ev?.steps?.length ? 'ONE-LINE SUMMARY' : 'MODEL RATIONALE'}
                note={ev?.steps?.length
                  ? 'RETURNED ALONGSIDE THE ANSWER'
                  : 'GENERATED AFTER THE CAUSE — A JUSTIFICATION, NOT THE REASONING'}
              >
                <div className="t-body" style={{ fontStyle: 'italic', lineHeight: 1.5 }}>
                  “{sample.rationale}”
                </div>
              </Section>
            )}

            <Section title="RESULT">
              <div className="t-micro" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                <span className="dim">AUTONOMOUS RESPONSE</span>
                <span className="acc">
                  {ACTION_WORD[sample.action] ?? sample.action.toUpperCase()}
                  {sample.target ? ` → ${sample.target}` : ''}
                </span>
              </div>
              <div className="t-micro" style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span className="dim">GROUND TRUTH</span>
                <span>
                  {CAUSE_WORD[sample.truth]}{' '}
                  {correct !== null && (
                    <span style={{ color: correct ? 'var(--accent)' : 'var(--alert)' }}>
                      {correct ? '✓' : '✗'}
                    </span>
                  )}
                </span>
              </div>
              <div className="t-micro faint" style={{ marginTop: 7, fontSize: 8, lineHeight: 1.5 }}>
                GROUND TRUTH IS WITHHELD FROM THE ADVISOR AT RUNTIME — IT IS CARRIED IN THE
                TRACE FOR SCORING ONLY.
              </div>
            </Section>
          </>
        )}
      </div>
    </div>
  )
}
