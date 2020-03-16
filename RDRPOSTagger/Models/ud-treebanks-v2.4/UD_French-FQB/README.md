# Summary

The corpus **UD_French-FQB** is an automatic conversion of the [French QuestionBank v1](http://alpage.inria.fr/Treebanks/FQB/), a corpus entirely made of questions

# Introduction

The original **French QuestionBank** is described in [Hard Time Parsing Questions: Building a QuestionBank for French.](https://hal.archives-ouvertes.fr/hal-01457184v2).
It was converted to **UD** with the conversion system described in the chapter 3 of the book [Application of Graph Rewriting to Natural Language Processing](https://hal.inria.fr/hal-01814386) and available on [Inria Gitlab](https://gitlab.inria.fr/grew/SSQ_UD).

The original annotation scheme versions (phrase-structure, surface dependencies following the FTB scheme, Deep syntax annotations following the [Deep Sequoia scheme](https://deep-sequoia.inria.fr) are available at the following [URL](http://alpage.inria.fr/Treebanks/FQB/).

# Recommended splits
Due to the UD constraints on the test set size (at least 10k tokens) , we recommand to simply concatenate this treebank to the [Sequoia](https://universaldependencies.org/treebanks/fr_sequoia/index.html) and [FTB](https://github.com/UniversalDependencies/UD_French-FTB) treebanks in order to get a robust, less domain sensitive, parser. Those 3 treebanks are perfectly compatible and were converted by the same team.

In our own experiments, we either used the **UD_French-FQB** in a 10-folds cross-validation scenario or in a train/dev/test scenario whith the  i_th sentence in train, i_th+1 in dev, i_th+2 in test.


# Statistics
* sentences: 2289
* words: 23236
* Average sentence lenght: 10.15

# Domains
* TREC 08-11: 1893 sents.
* French Government/NGOs FAQs: 196 sents.
* CLEF 03: 200 (sents.)

Note that the TREC domain questions are a translation of the corresponding questions in the English Question Bank (Judge et al, 2006). 



# Acknowledgments
* contributors: Marie Candito, Bruno Guillaume, Djamé Seddah
* contact: Djamé Seddah: djame.seddah@paris-sorbonne.fr, Marie Candito: marie.candito@linguist.univ-paris-diderot.fr
* UD maintainer: Bruno Guillaume, bruno.guillaume@loria.fr


## References

* Djamé Seddah, Marie Candito. [Hard Time Parsing Questions: Building a QuestionBank for French](https://hal.archives-ouvertes.fr/hal-01457184). Tenth International Conference on Language Resources and Evaluation (LREC 2016), May 2016, Portorož, Slovenia.

* Guillaume Bonfante, Bruno Guillaume, Guy Perrier. [Application of Graph Rewriting to Natural Language Processing](https://hal.inria.fr/hal-01814386). ISTE Wiley, 1, pp.272, 2018, Logic, Linguistics and Computer Science Set, Christian Rétoré, 1786300966. ⟨hal-01814386⟩

* John Judge, Aoife Cahill,  and Joseph van Genabith,  (2006). [QuestionBank: Creating a Corpus of Parse-Annotated Questions](https://aclweb.org/anthology/papers/P/P06/P06-1063/). In Proceedings of the 21st International Conference on Computational Linguistics and 44th Anual Meeting of the Association for Computational Linguistics (COLING-ACL 2006), pages 497–504, Sydney, Australia.



<pre>
=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: UD v2.4
License: LGPL-LR
Includes text: yes
Genre: nonfiction news
Lemmas: converted from manual
UPOS: converted from manual
XPOS: manual native
Features: converted from manual
Relations: converted from manual
Contributors: Seddah, Djamé; Candito, Marie; Guillaume, Bruno
Contributing: elsewhere
Contact: djame.seddah@gmail.com
===============================================================================
</pre>
