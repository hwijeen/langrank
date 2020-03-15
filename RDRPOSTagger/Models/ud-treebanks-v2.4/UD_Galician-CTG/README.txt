# Summary

The Galician UD treebank is based on the automatic parsing of the Galician Technical Corpus (http://sli.uvigo.gal/CTG) created at the University of Vigo by the the TALG NLP research group.

# Introduction

Original corpus sentences were selected and shuffled at random, and divided in 60-20-20 splits for the train, dev and test files, respectively.

The bootstrap version of the CTG UD annotated corpus was obtained by using FreeLing 4.0 parser with the Treeler library, and by adapting the POS and dependency relations tags to CoNLL-U Format. Next versions of the corpus imply a review of the results of this initial version.

The Galician UD treebank covers mainly technical texts of the fields of medicine, sociology, ecology, economy and law.

# Stats (UD version 2.1)

* 3993 sentences
* 126011 tokens
* 138837 syntactic words

# Changelog

* 2018-04-15 v2.2
  * Repository renamed from UD_Galician to UD_Galician-CTG.
* 2017-10-23 UD release 2.1:
 * Added multiword tokens for contractions
 * Corrected errors in sentence segmentation
 * Several manual and automatic fixes in corpus tagging and encoding
 * Added raw text

# Acknowledgments

* Special thanks to Martin Popel and Dan Zeman for their invaluable help

# Reference

* Gómez Guinovart, Xavier (2017): Recursos integrados da lingua galega para a investigación lingüística. Marta Negro Romero, Rosario Álvarez, Eduardo Moscoso Mato (eds.): Gallæcia. Estudos de lingüística portuguesa e galega (ISBN 978-84-16954-41-4), Universidade de Santiago de Compostela, Santiago de Compostela, pp. 1045-1056. DOI: http://dx.doi.org/10.15304/cc.2017.1080.61



<pre>
=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: UD v1.3
License: CC BY-NC-SA 3.0
Includes text: yes
Genre: medical legal nonfiction news
Lemmas: converted from manual
UPOS: converted from manual
XPOS: manual native
Features: not available
Relations: converted from manual
Contributors: Gómez Guinovart, Xavier
Contributing: elsewhere
Contact: xgg@uvigo.es
===============================================================================
</pre>
