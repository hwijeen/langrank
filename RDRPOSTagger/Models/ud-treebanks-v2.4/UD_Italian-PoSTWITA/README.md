# Summary

PoSTWITA-UD is a collection of Italian tweets annotated in Universal Dependencies that can be exploited for the training of NLP systems to enhance their performance on social media texts.


# Introduction

PoSTWITA-UD has been created by enriching the dataset used for the EVALITA 2016 task of Part-of-Speech tagging of Social Media (see (Bosco et al. 2016)).
The original corpus consists of 6,438 tweets of the development set (114,967 tokens) and 300 tweets of the test set (4,759 tokens), annotated at PoS level only. 
The conversion and syntactic annotation process was carried out through alternating steps of automatic scripting and manual revision, and finally with some out-of-domain parsing experiments. Parsing results also underwent a manual revision by two independent annotators.
 
**New since v2.3**: In order to meet the requirements of the EU General Data Protection Regulation (GDPR), entered into force on May 2018, the resource content has been pseudonymized, by substituting original tweet IDs and user names.

# Acknowledgements

The treebank development has also been possible thanks to the contribution of Fabio Tamburini (University of Bologna), who used AnIta for lemmatization and morphological analysis, and of Oronzo Antonelli (University of Bologna), Alberto Lavelli (FBK, Trento), and Alessandro Mazzei (University of Turin), who worked on the parsing experiments.


# Corpus splitting

The treebank has been randomly split as follows:
* it_postwita-ud-train.conllu: 5,368 tweets (99,441 words)
* it_postwita-ud-dev.conllu: 671 tweets (12,335 words)
* it_postwita-ud-test.conllu: 674 tweets (12,668 words)

:warning: The current partition of the resource is different from the version used for the EVALITA campaign in 2016.


# Basic statistics

Tree count: 6,712
Word count: 124,410
Token count: 119,238
Dep. relations: 53 of which 18 language-specific
POS tags: 16
Category=value feature pairs: 35


# References

* Manuela Sanguinetti, Cristina Bosco, Alberto Lavelli, Alessandro Mazzei, Fabio Tamburini. 2018. PoSTWITA-UD: an Italian Twitter Treebank in Universal Dependencies. Accepted paper at LREC 2018

* Manuela Sanguinetti, Cristina Bosco, Alessandro Mazzei, Alberto Lavelli, Fabio Tamburini. 2017. Annotating Italian Social Media Texts in Universal Dependencies. In: Proceedings of the Fourth International Conference on Dependency Linguistics (Depling 2017), Pisa (Italy), pp. 229–239

* Cristina Bosco, Fabio Tamburini, Andrea Bolioli, Alessandro Mazzei. 2016. Overview of the EVALITA 2016 Part Of Speech on TWitter for ITAlian task. In: Proceedings of Evalita 2016

# Changelog
2019-11-15 v2.4
* various corrections to pass validation

2018-11-15 v2.3
* substituted original tweet IDs and user names
* minor corrections

2018-04-15 v2.2 
* added new data
* added new label subtypes		
* more systematic treatment of dates, predeterminers, coordination, pre-verbal objects, and of various discourse markers
* revised parataxis:insert
* other minor corrections

2017-11-15 v2.1
* initial release

# Metadata

=== Machine-readable metadata =================================================

```
Data available since: UD v2.1
License: CC BY-NC-SA 4.0
Includes text: yes
Genre: social
Lemmas: automatic with corrections
UPOS: converted with corrections
XPOS: automatic with corrections
Features: automatic with corrections
Relations: automatic with corrections
Contributors: Bosco, Cristina; Sanguinetti, Manuela
Contributing: elsewhere
Contact: msanguin@di.unito.it
```
===============================================================================
