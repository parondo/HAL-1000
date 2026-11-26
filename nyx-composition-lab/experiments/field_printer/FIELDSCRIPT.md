# FieldScript — draft 0.1

Declarative description of typed field composition. The example is human/AI-readable design notation, **not an implemented parser or executable language**.

Semantics for P0: REFERENCE selects a finite atlas and target; FIELD declares unit amplitude and a relative exponent in J17; COMPOSE adds the three cyclotomic terms; ACTIVATE selects equality of the quadratic trace to 9; VERIFY requests the listed finite checks. The Python script implements the specified example directly.

Future operations TRANSPORT, RECAPTURE, and STOP need versioned typed contracts. Undefined operations are not zero and must not silently become false. No command may target hardware in this prototype. A future compiler must reject missing calibration or unapproved physical bridges before generating instrument commands.
