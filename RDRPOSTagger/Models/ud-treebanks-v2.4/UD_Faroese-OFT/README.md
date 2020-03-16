# Summary

This is a treebank of Faroese based on the Faroese Wikipedia.

# Introduction

The treebank is based on sentences from the Faroese Wikipedia. The whole
Wikipedia was analysed using Trond Trosterud's tools for Faroese.[1] We
took all the sentences and discarded those with unknown words.

The remaining sentences were manually annotated for Universal Dependencies
and the morphology and POS tags were converted deterministically using
a lookup table. Errors in the original morphology and disambiguation 
were corrected where found.

The treebank contains a lot of copula sentences and very little first
or second person, as can be expected from Wikipedia texts.

1. http://gtweb.uit.no/cgi-bin/smi/smi.cgi?text=%C3%81+tunguni+eru+sm%C3%A1ar+tenn.&action=analyze&lang=fao&plang=eng

# Changelog

* 2018-11-15 v1.0
  * First release in UD

# Acknowledgements

The morphology and preliminary disambiguation was done by Trond Trosterud's 
finite-state morphology and constraint grammar for Faroese.

If you use this treebank in your work, please cite:

```
@inproceedings{tyersetal18-faroese,
    author = {Francis M. Tyers and Mariya Sheyanova and Alexandra Martynova and Pavel Stepachev and Konstantin Vinogradovsky},
    title = {Multi-source synthetic treebank creation for improved cross-lingual dependency parsing},
    booktitle = {Proceedings of the Second Workshop on Universal Dependencies (UDW 2018)},
    pages = {144--150},
    year = 2018
}
```

<pre>
=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: UD v2.2
License: CC BY-SA 4.0
Includes text: yes
Genre: wiki 
Lemmas: converted from manual
UPOS: converted from manual
XPOS: manual native
Features: converted from manual
Relations: manual native
Contributors: Zeman, Daniel; Mortensen, Bjartur; Tyers, Francis
Contributing: elsewhere
Contact: ftyers@hse.ru
===============================================================================
</pre>
