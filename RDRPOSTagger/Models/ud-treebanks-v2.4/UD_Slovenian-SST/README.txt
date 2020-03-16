# Summary

The Spoken Slovenian UD Treebank (SST) is the first syntactically annotated corpus of spoken Slovenian, based on a sample of the reference GOS corpus, a collection of transcribed audio recordings of monologic, dialogic and multi-party spontaneous speech in different everyday situations.

# Introduction

The Spoken Slovenian UD Treebank (SST) is the first syntactically annotated corpus of spoken Slovenian. The manual annotations have been performed on a representative sample of the reference Gos corpus of spoken Slovenian (Zwitter Vitez et al. 2013), a collection of transcribed audio recordings of monologic, dialogic and multi-party spontaneous speech in different everyday situations. The treebank is representative of speakers (sex, age, region, education), communication channels (TV, radio, telephone, personal contact) and communication settings (TV and radio shows, lectures, meetings, consultations, services, conversations between friends etc.).

The spelling, tokenization and segmentation principles follow the transcription guidelines of the reference corpus (Verdonik et al. 2013) with the syntactic trees spanning over individual utterances (semantically, syntactically and acoustically delimited units, roughly corresponding to written sentences). The morphological and syntactic analysis in the SST treebank has been performed on top of normalized transcriptions that reduce the number of token types due to regional, colloquial and other pronunciation variation. However, information on pronunciation-based transcription and tokenization has been included as part of the MISC column.

The SST treebank includes manual annotations of lemmas, part-of-speech categories, morphological features and dependency relations in accordance with the Universal Dependencies annotation scheme. In addition to language-specific extensions, in line with the written Slovenian UD Treebank, the SST treebank also includes new speech-specific extensions to accommodate the structural and pragmatic particularities of spoken language syntax, such as disfluencies, fillers, parentheticals, general extenders etc.  More information on the treebank construction and annotation is given in Dobrovoljc and Nivre (2016).


The current version of the SST treebank includes 3,188 utterances (sentences) or 29,488 tokens, produced by 606 speakers in 287 different speech events. As opposed to previous releases with unequal genre distributions, sentence-level randomization and different train-dev-test splits (UDv1 to UDv2.1), the SST UDv2.2 data has been randomized on text-level and split into testing (10,015) and training (19,473) subsets, in accordance with the CONLL-ST 2018 requirements. The original order of the utterances can be restored by sentence IDs.

# Acknowledgments

Kaja Dobrovoljc (treebank construction and annotation)
Joakim Nivre (guidelines consulting)


# References

* Kaja Dobrovoljc and Joakim Nivre. 2016. The Universal Dependencies Treebank of Spoken Slovenian. In: Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16), Portorož, Slovenia.
* Darinka Verdonik, Iztok Kosem, Ana Zwitter Vitez, Simon Krek and Marko Stabej. 2013. Compilation, transcription and usage of a reference speech corpus: the case of the Slovene corpus GOS. Language Resources and  Evaluation, 47(4):1031–1048.
* Ana Zwitter Vitez, Jana Zemljarič Miklavčič, Simon Krek, Marko Stabej and Tomaž Erjavec. 2013. Spoken corpus Gos 1.0. Slovenian language resource repository CLARIN.SI. http://hdl.handle.net/11356/1040.


# Changelog

2015-01-30 v2.0
    * Manual and automatic conversions from UDv1 to UDv2 guidelines
    * Manual corrections of some mistakes in previous versions
    * Resizing of train-dev-test (in accordance with CONLL ST 2017 requirements)
    * Random utterance shuffling to ensure more representative genre distributions.
    
2015-03-15 v2.2
    * Manual corrections of some mistakes in previous versions
    * New (text-level) data randomization
    * Resizing of train-test datasets (in accordance with CONLL ST 2018)


=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: UD v1.3
License: CC BY-NC-SA 4.0
Includes text: yes
Genre: spoken
Lemmas: manual native
UPOS: manual native
XPOS: manual native
Features: manual native
Relations: manual native
Contributors: Dobrovoljc, Kaja; Nivre, Joakim
Contributing: elsewhere
Contact: kaja.dobrovoljc@gmail.com
===============================================================================
