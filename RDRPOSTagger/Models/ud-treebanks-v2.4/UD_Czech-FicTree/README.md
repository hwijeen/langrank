# Summary

FicTree is a treebank of Czech fiction, automatically converted into the UD
format. The treebank was built at Charles University in Prague.


# Introduction

The treebank consists of 12,760 sentences (166,432 tokens). The texts come from
eight literary works published in the Czech Republic between 1991 and 2007.
The text data was manually annotated according to the Prague Dependency Treebank
guidelines, then converted into the UD format.
To comply with agreements concluded with the copyright holders, the texts are
shuffled into random chunks of maximum 100 words.
The treebank is licensed under the terms of [CC BY-NC-SA 3.0]
(http://creativecommons.org/licenses/by-nc-sa/3.0/).


# Annotation and conversion

The texts were parsed independently by two parsers trained on the Prague
Dependency Treebank data (analytical layer). The parsing results were manually
corrected and the two versions merged. Any differences were resolved manually.
The morphological and syntactic annotation according to the UD guidelines
was assigned by converting the original PDT annotation.
The conversion procedure was designed by Dan Zeman and implemented in Treex.


# Acknowledgments

We wish to thank the participants in the annotation effort, including Milena
Hnátková, Tomáš Jelínek, Ivana Klímová, Alena Kropíková, Hana Skoumalová and
Olga Zitová; as well as Dan Zeman for the data conversion.


## References

* Tomáš Jelínek. 2017. FicTree: a Manually Annotated Treebank of Czech Fiction.
 In: J. Hlaváčová (Ed.): ITAT 2017 Proceedings, pp. 181–185.
 http://ceur-ws.org/Vol-1885/181.pdf


# Text details

The eight texts in the treebank include six fiction titles, a children’s fiction
book, and a book of memoirs.
Most of the texts were first published between 1991 and 2007 except one text,
published in 1969.
80% of the texts are original Czech texts, 20% are translations (from German and
Slovak).


# Changelog

* 2019-05-01 v2.4
  * Modified conversion: nouns do not have objects.
  * Unknown tag with advmod --> ADV.
  * Fixed punctuation attachment.
* 2018-11-15 v2.3
  * Split multi-word tokens "cos, ses, sis, tys, vždyťs", participle + "-s".
  * Bug fix: conditional "by" should be attached as 'aux', not 'aux:pass'.
  * Added NameType, better classification of proper names. Consequence: many places now analyzed as 'flat'.
* 2018-04-15 v2.2
  * Added enhanced representation of dependencies propagated across coordination.
    The distinction of shared and private dependents is derived deterministically from the original Prague annotation.
* 2017-11-15 v2.1
  * First official release.


<pre>
=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: UD v2.1
License: CC BY-NC-SA 4.0
Includes text: yes
Genre: fiction
Lemmas: converted from manual
UPOS: converted from manual
XPOS: manual native
Features: converted from manual
Relations: converted from manual
Contributors: Jelínek, Tomáš; Zeman, Daniel
Contributing: elsewhere
Contact: tomas.jelinek@ff.cuni.cz, zeman@ufal.mff.cuni.cz
===============================================================================
</pre>
