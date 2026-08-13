# Meta-factorization

We introduce meta-factorization, a theory that describes matrix decompositions as solutions of linear matrix equations: the projector and the reconstruction equation. Meta-factorization reconstructs known factorizations, reveals their internal structures, and allows for introducing modifications, as illustrated with SVD, QR, and UTV factorizations. The prospect of meta-factorization also provides insights into computational aspects of generalized matrix inverses and randomized linear algebra algorithms. The relations between the Moore-Penrose pseudoinverse, generalized Nystroem method, and the CUR decomposition are revealed here as an illustration. Finally, meta-factorization offers hints on the structure of new factorizations and provides the potential of creating them.

## Contents

- `mft.m` — the meta-factorization routines illustrating the constructions of the paper.
- `network_example.m` — the worked example of Section 4 (factor replacement on a resistor
  network). It is **byte-for-byte the listing printed in the paper**, plus a header. Running it
  reproduces the two output blocks displayed there, in order:

  ```
  rank 4   test 3.114894e-14   err 3.752010e-14   det  1.000
  rank 3   test 3.291403e+00   err 3.415650e+00   det -0.000
  rank 5   test 3.044879e-14   err 4.435485e-14   det  1.000
  rank 4   test 6.895672e-15   err 1.815507e-14   det  0.000
  ```

  It is self-contained: no toolbox, no other file, no random stream. Row 2 is the refused
  candidate; its criterion norm is exactly `sqrt(390)/6` and its reconstruction residual exactly
  `sqrt(105)/3`. Every core is built with `pinv`, as printed — building the augmented (`k = m`)
  core with left/right division instead is algebraically equivalent but changes the roundoff
  digits, so the two must not be mixed.

## Reference Paper


Karpowicz, M. P. (2021). A theory of meta-factorization. arXiv preprint arXiv:2111.14385.
```
@misc{karpowicz2021theory,
      title={A theory of meta-factorization}, 
      author={Michał P. Karpowicz},
      year={2021},
      eprint={2111.14385},
      archivePrefix={arXiv},
      primaryClass={math.NA}
}
