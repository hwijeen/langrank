# Summary

UD_Old_French-SRCMF is a conversion of (part of) the SRCMF corpus (Syntactic Reference Corpus of Medieval French [srcmf.org](http://srcmf.org/)).

# Introduction

UD_Old_French-SRCMF consists in 10 texts spanning from 9th to 13th century. It includes 17678 sentences and 170 741 tokens.

Sentences are annotated with the following metadata :
+ sent_id : a unique id for each sentence in the treebank
+ text : the sentence
+ newdoc id : a unique id for each of the texts. This id can be split on underscores to get back :
    + name of the text
    + date
    + form : verse and/or prose


The following table lists the texts used in this treebank :

| ID | Author | Name of the text | Number of tokens |
| ------ | ------ | ------ | ------ |
| Strasbourg_842_prose | anonymous | Serments de Strasbourg | 115 |
| StLegier_1000_verse | anonymous | Vie de saint Léger | 1,388 |
| StAlexis_1050_verse | anonymous | Vie de saint Alexis | 4,750 |
| Roland_1100_verse | anonymous | Chanson de Roland | 28,752 |
| Lapidaire_mid12_prose | anonymous | Lapidaire en prose | 4,708 |
| QuatreLivresReis_late12_prose | anonymous | Quatre livres des reis | 12,949 |
| BeroulTristan_late12_verse | Beroul, Tristan | Tristan de Beroul  | 26,766 |
| TroyesYvain_1180_verse | Chrestien de Troyes, Yvain | Yvain de Chretien de Troyes | 41,256 |
| Aucassin_early13_verse-prose | anonymous | Aucassin et Nicolet | 9,838 |
| Graal_1225_prose | anonymous | Queste del Saint Graal | 40,219 |


# Structure

The corpus results from a conversion from the SRCMF corpus.

In the SRCMF projet, texts with less than about 40,000 words were entirely annotated, while texts with more than 40,000 words were sampled in three parts (beginning, middle and end of the text) to reach a total amount of about 40000 words.

As a result, UD_Old_French-SRCMF includes 8 full texts (Strasbourg, StLegier, StAlexis, Roland, Lapidaire, BeroulTristan, TroyesYvain, Aucassin) and 2 sampled ones (QuatreLivresReis and Graal).

The treebank is split as follows (in number of tokens) :

| ID | Train | Test | Dev |
| ------ | ------ | ------ | ------ |
| Strasbourg_842_prose | 115 | 0 | 0 |
| StLegier_1000_verse | 1388 | 0 | 0 |
| StAlexis_1050_verse | 4750 | 0 | 0 |
| Roland_1100_verse | 18752 | 4999 | 5001 |
| Lapidaire_mid12_prose | 0 | 2361 | 2347 |
| QuatreLivresReis_late12_prose | 12949 | 0 | 0 |
| BeroulTristan_late12_verse | 16748 | 5001 | 5017 |
| TroyesYvain_1180_verse | 41256 | 0 | 0 |
| Aucassin_early13_verse-prose | 9838 | 0 | 0 |
| Graal_1225_prose | 30225 | 4997 | 4997 |
| Total | 136021 | 17358 | 17362 |

It is to be noticed that most of Dev and Test data is taken from texts that also exist in Training data, but these are large texts, so it does not make the testing too easy…
One text (Lapidaire) is represented only in Dev/Test but not in train.
Pre-1100 texts are only in Train, because they seem too small to reserve anything for testing.
The rest of the data (136 021 /170 741 tokens) is for Train corpus.


# Deviations from UD

We added some more specific relations (subtypes), either to specify a relation, or in the case of tokens entering a double dependency relation (typically : relative pronouns and  contracted forms) :

+ `acl:relcl` : relative clause
+ `advmod:obl` : contracted `advmod` + `obl` (eg. _sin_ = _si_ + _en_)
+ `aux:pass` : passive auxiliary
+ `case:det` : contracted `case` + `det` (eg. _del_ = _de_ + _le_)
+ `cc:nc` : non coordinating conjunction (eg. _et_ at the beginning of a sentence)
+ `mark:advmod` : `mark` and `advmod` (eg. _coment_ at the beginning of a subordinate clause)
+ `mark:obj` : `mark` and `obj` (eg. relative pronoun _que_)
+ `mark:obl` : `mark` and `obl` (eg. relative pronoun _cui_ / _qui_ )
+ `nsubj:advmod` : contracted `nsubj` + `advmod` (eg. _jon_ = _jo_ + _en_)
+ `nsubj:obj` : contracted `nsubj` + `obj` (eg. _quil_ = _qui_ + _le_)
+ `obj:advmod` : contracted `advmod` + `obj` (eg. _sis_ = _si_ + _les_)
+ `obj:advneg` : contracted `negation` + `obj` (eg. _nes_ = _ne_ + _les_)
+ `obj:obl` : contracted `obl` + `obj` (eg. _oul_ = _ou_ + _le_)
+ `obl:advmod `: the double labelling accounts for the difficulty to decide between obl and advmod relations (`en` and `i`).

We added some features :

+ Morph=VFin : finite verb
+ Morph=VInf : non finite verb
+ Morph=VPar : verbal participle
+ Polarity=Int : interrogative
+ PronType=Ord : numeral

Consult [the language specific documentation](http://universaldependencies.org/fro/dep/index.html) for further details concerning subtypes.


# Acknowledgments


UD_Old_French-SRCMF results from the conversion of (part of) the SRCMF corpus (Syntactic Reference Corpus of Medieval French [srcmf.org](srcmf.org)).

This conversion was achieved by Aurélie Collomb, in the frame of a internship funded by lab Lattice (Paris, CNRS, ENS & Université Sorbonne Nouvelle Paris 3, PSL & USPC), and supervised by Sophie Prévost, Isabelle Tellier and Kim Gerdes. Marine Courtin achieved the deposit of the files, and especially took in charge the validation of the corpus through the successive steps of the process.

The SRCMF corpus results from the SRCMF project which took place in 2008-2012, funded by the ANR (France) and the DFG (Germany), and supervised by Sophie Prévost and Achim Stein.

The SRCMF project consisted in the manual syntactic annotation of 15 texts (251,000 tokens) from the 9th to 13th C. Part-of-speech tags were for most of them retrieved from the already existing tagging of the texts (stemming from: Base de Français Medieval, Lyon, ENS de Lyon, IHRIM Laboratory [http://txm.bfm-corpus.org]([http://txm.bfm-corpus.org]), and the Nouveau Corpus d'Amsterdam [http://www.uni-stuttgart.de/lingrom/stein/corpus#nca]([http://www.uni-stuttgart.de/lingrom/stein/corpus#nca]))

The contributors to the SRCMF project were: Stein, Achim; Prévost, Sophie; Rainsford, Tom; Mazziotta, Nicolas;  Bischoff Béatrice; Glikman, Julie; Lavrentiev, Alexei; Heiden, Serge; Guillot-Barbance, Céline; Marchello-Nizia, Christiane.

The conversion from the original SRCMF annotation to the SRCMF-UD annotation was done automatically both for the POS and the syntactic relations, thanks to a set of elaborated rules.
Some 1,200 syntactic relations left unlabelled were then manually annotated (Sophie Prévost), and significant spot-checking occurred, focusing on potential difficulties (eg. conj relation).

The whole SRCMF corpus (251,000 tokens) was actually automatically converted into UD dependencies, but only 172,000 tokens have so far undergone a significant checking: the remaining 80,000 tokens will be added to UD_Old_French-SRCMF for the next release.


## References

* Stein, A. et Prévost, S. 2013. Syntactic annotation of medieval texts : the Syntactic Reference Corpus of Medieval French (SRCMF). In P. Bennett, M. Durrell, S. Scheible and R. Whitt (éds) New Methods in Historical Corpus Linguistics, Corpus Linguistics and International Perspectives on Language, CLIP Vol. 3. Tübingen: Narr., p. 75-82. [halshs-01122079]

# Changelog

* 2018-04-15 v2.2
  * Initial release

<pre>
=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: UD v2.2
License: CC BY-NC-SA 3.0
Includes text: yes
Genre: nonfiction legal poetry
Lemmas: not available
UPOS: converted with corrections
XPOS: manual native
Features: automatic
Relations: automatic with corrections
Contributors: Prévost, Sophie; Collomb, Aurélie; Gerdes, Kim; Tellier, Isabelle; Courtin, Marine; Lavrentiev, Alexei; Guillot-Barbance, Céline
Contributing: here source
Contact: sophie.prevost@ens.fr
===============================================================================
</pre>
