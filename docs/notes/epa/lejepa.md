# LeJEPA

[Balestriero and LeCun, 2025 — LeJEPA](https://arxiv.org/html/2511.08544v1)

LeJEPA brings vectors from different views of an input together while SIGReg
keeps the overall distribution spread out. SIGReg compares random
projections with a Gaussian distribution. This resists the easy solution
where every input gets the same vector, without an EMA teacher or
stop-gradient.

That is why it appears behind HEPA and LeNEPA. It changes how training avoids
collapse; it does not, by itself, make the target a future event.

It could help us test whether a teacher is necessary. We would still need
sensible traffic views and evidence that rare attack details survive. A
spread-out representation is not automatically a useful detector. There is
no separate LeJEPA rung planned.
