# Summary

The Croatian UD treebank is based on the extension of the SETimes-HR corpus, the [hr500k](http://hdl.handle.net/11356/1183) corpus.


# Introduction

The sentences are partially parallel with the smaller Serbian UD treebank, which
comes from the Serbian edition of SETimes. For the UD release 2.4, the Croatian and Serbian 
corpus were enriched with ```newdoc``` metadata and re-split so that the corresponding
documents are in the same section (train/dev/test) in Croatian and Serbian.

Sentence ids have also been changed to reflect the domain / source the data comes from,
and not the dependence to one of the sections (train/dev/test), as was the case
in previous releases. 

### Training set.

Contains 6,844 sentences (151,226 tokens) from three sources:

1. sentence ids ```set.hr*```: pseudorandom 80% of documents of newspaper text from the [Southeast European Times](http://en.wikipedia.org/wiki/Southeast_European_Times) news website, obtained from the [SETimes parallel corpus](http://nlp.ffzg.hr/resources/corpora/setimes/). This part of the treebank is built on top of the [SETimes.HR dependency treebank of Croatian](https://github.com/ffnlp/sethr);
2. sentence ids ```news.hr*```: pseudorandom 80% of documents of Croatian news web sources.
3. sentence ids ```web.hr*``` : pseudorandom 80% of sentences of [Croatian web sources](http://nl.ijs.si/isjt14/proceedings/isjt2014_10.pdf).

### Development set.

Contains 954 sentences (21,952 tokens) from three sources:

1. sentence ids ```set.hr*```: pseudorandom 10% of documents of newspaper text from the Southeast European Times.
2. sentence ids ```news.hr*```: pseudorandom 10% of documents of Croatian news web sources.
3. sentence ids ```web.hr*``` : pseudorandom 10% of sentences of Croatian web sources.

### Test set.

Contains 1214 sentences (26,263 tokens) from four sources:

1. sentence ids ```set.hr*```: pseudorandom 10% of documents of newspaper text from the Southeast European Times (+ the previous test set from the same source).
2. sentence ids ```wiki.hr*```: old Wikipedia-based test set.
2. sentence ids ```news.hr*```: pseudorandom 10% of documents of Croatian news web sources.
3. sentence ids ```web.hr*``` : pseudorandom 10% of sentences of Croatian web sources.

### Details

Sentence and word segmentation was manually checked. The treebank does not include multiword tokens. No language-specific features and relations were used. The POS tags and features were converted from [Multext East v6](http://nl.ijs.si/ME/V6/msd/html/msd-hbs.html) (present in the XPOS column) and manually checked. The syntactic annotation was done manually.

# Acknowledgments

When using the Croatian UD treebank, please cite the following paper:

* Željko Agić and Nikola Ljubešić. 2015. [Universal Dependencies for Croatian (that work for Serbian, too).](http://aclweb.org/anthology/W/W15/W15-5301.pdf). In Proc. BSNLP, pp. 1--8 ([bib](http://aclweb.org/anthology/W/W15/W15-5301.bib)).

See file LICENSE.txt for further licensing information.


# Changelog

* 2018-04-30 v2.4
  * Sentences renamed so that not the old split is encoded, but the source of the data
  * Data split made compatible on the document level with the parallel data in UD_Serbian-SET.

* 2018-04-15 v2.2
  * Repository renamed from UD_Croatian to UD_Croatian-SET.
  * Data split made compatible with the parallel data in UD_Serbian-SET.
* 2017-02-15
  * converted to UD v2 standard
    * nmod vs. obl under non-verbal predicates should be checked manually (see the ToDo attribute in the MISC column)
    * by UD guidelines, reflexive pronouns with inherently reflexive verbs are now attached as expl:pv, not compound
    * adverbial participles (converbs) are marked by VerbForm=Conv
  * a number of enhancements and bug fixes
    * all pronouns, determiners and pronominal adverbs have PronType
    * all verbs have VerbForm; all finite verbs have Mood
    * ordinal numerals are ADJ like elsewhere in UD, not NUM (but they keep NumType=Ord)
    * relative pronouns, determiners and adverbs are not attached as mark (subordinating conjunctions keep the mark relation)
    * possessive adjectives and determiners are amod and det respectively; not nmod
    * coordinating conjunctions at the beginning of sentence are attached as cc, not discourse
* 2017-02-09
  * added new ud v1 sentences from news-hr to dev, test, and train set: 2600 sentences, out of which the last 703 went to dev (400) and test (303), and the remainder to the train set
* 2016-10-31
  * added 2235 new sentences to the training set, and 97 new sentences to the test set, from various Croatian web sources



<pre>
=== Machine-readable metadata =================================================
Data available since: UD v1.1
License: CC BY-SA 4.0
Includes text: yes
Genre: news web wiki
Lemmas: converted from manual
UPOS: converted from manual
XPOS: manual native
Features: converted from manual
Relations: manual native
Contributors: Agić, Željko; Ljubešić, Nikola; Zeman, Daniel
Contributing: elsewhere
Contact: zeljko.agic@gmail.com
===============================================================================
</pre>
