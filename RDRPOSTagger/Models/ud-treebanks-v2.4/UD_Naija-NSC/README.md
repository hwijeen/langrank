# Summary

A Universal Dependencies corpus for spoken Naija (Nigerian Pidgin).

# Introduction

The corpus is based on dialogues and monologues and comprises 948 sentences and 12863 tokens.

Sentences are annotated with the following metadata :
+ sent_id (which also indicates the file)
+ text
+ text_en (English translation)
+ speaker (for dialogues).

# Structure

The text has been transcribed mostly following English spelling conventions for lexical words. Grammatical words have been transcribed following consensual conventions elaborated by the annotators.


The text is segmented into illocutionary units. The end of illocutionary units is indicated by a double slash (//). The sentence nucleus containing the predicate is separated from dislocated units by "lesser than" signs (<) from left-dislocated elements, and by "greater than" signs (>) from right-dislocated units. Paradigmatic lists (coordinations, appositions, and disfluencies) are marked with curly breackets, each conjunct being separated by the pipe symbol (|). Further details can be found on the "Macrosyntactic annotation guide".

# Deviations from UD

- We distinguish arguments, modifiers, and peripherals in the `obl` relation: `obl:arg`, `obl:mod` `obl:periph`
- We used `conj:coord` instead of `coord`.
- We used `conj:dicto` instead of `reparandum`.
- We used `conj:appos` instead of `appos`.
- We use `compound:redup` for reduplications and `compound:svc` for serial verb constructions.
- We distinguish 6 types of parataxis: `parataxis:conj`, `parataxis:discourse`, `parataxis:dislocated`, `parataxis:insert`, `parataxis:obj`, `parataxis:parenth`.

Consult [the language specific documentation](http://universaldependencies.org/pcm/dep/index.html) for further details concerning subtypes.


# Acknowledgments

The treebank was created within the NaijaSynCor project, directed by Bernard Caron and funded by the ANR, the French National Research Agency.

This corpus is a pilot for the larger corpus elaborated as part of the NaijaSynCor Project (Projet-ANR-16-CE27-0007). Its main aim is to elaborate and test the annotation and procedures that are used in the ANR-project. It will be part of a larger 500kW corpus that will be projected on prosodic and information structures and analysed for sociolinguistics variation (http://naijasyncor.huma-num.fr/).

The pilot corpus was recorded in various locations in Ibadan (Nigeria) by Bukola Babalola and Opeyemi Lewis. It was transcribed, translated and tagged manually using Elan-Corpa (http://llacan.vjf.cnrs.fr/res_ELAN-CorpA_en.php) by Folakemi Ladoja, Emeka Onwuegbuzia, Biola Oyelere and Samson Tella under the supervision of Bernard Caron. It was converted to CONLL by Mourad Aouini. The final Universal dependencies annotations have been manually checked by Sandra Bello, Marine Courtin, Bernard Caron, Kim Gerdes, Sylvain Kahane, and Manying Zhang using the processing chain developed by Kim Gerdes. The guidelines were written by Marine Courtin and Sandra Bellato under the supervision of Sylvain Kahane, Bernard Caron, and Kim Gerdes.

# Changelog

* 2019-05-15 v2.4
  * Modifications to comply with v2.4 validation:
    * when we had to change the POS tag to comply with UD validation rules, we kept the old POS tag in XPOS, so that no information is lost
    * in case of non-projective punctuations, we attached them on the govenor of the arc that creates the non-projectivity
    * compound:redup become conj:redup
* 2018-11-15 v2.3
* 2018-04-15 v2.2
  * Initial release


=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: UD v2.2
Includes text: yes
License: CC BY-SA 4.0
Genre: spoken
Lemmas: automatic
UPOS: manual native
XPOS: not available
Features: not available
Relations: manual native
Contributors: Caron, Bernard; Courtin, Marine; Gerdes, Kim; Kahane, Sylvain; Bellato, Sandra; Zhang, Manying
Contributing: here source
Contact: kim@gerdes.fr
===============================================================================
