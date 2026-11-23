"""Design-code formulas, turned into the curves and numbers the solver reads.

THE SOLVER KNOWS NO CODE. It reads physical laws stated as
data -- a tendon's relaxation loss against time, a creep coefficient
against age, a shrinkage strain against time -- and a case file carries
those numbers, so a case gives the same answer whatever becomes of the
formulas that once produced it. The formulas live here, one module per
code, used by the translators (a Midas model says which code it was built
to, and the translation does what that code says) and by anyone who wants
a code's curve for their own case. A new edition is a new module; an old
one stays, so an old model can be translated as it was.

Each formula is checked against the output of the program the models come
from, not against memory: see each function's docstring for what it was
checked against.
"""
