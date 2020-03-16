# Summary

It aims at gathering texts of the German literary history. Currently, it hosts Fragments of the early Romanticism, i.e. aphorism-like texts mainly dealing with philosophical issues concerning art, beauty and related topics.

# Introduction

In a long-term perspective, this treebank aims at gathering texts from different genres and different authors of the German literary history. Currently, it exclusively hosts Fragments of the early Romanticism (end of the 18th century, modern German), i.e. really short texts, often in aphorism-like form, dealing with philosophical issues in a witty and cryptic way. They mainly deal with aesthetics, i.e. with philosophy concerning art and beauty.
This treebank is mainly intended for stylistic analysis that can benefit from the dependency formalism as well as from the opportunity to automatically and quickly retrieve information concerning syntax.

Version 2.4 hosts the following texts (each text is followed by the reference to the original edition from which it  was digitized, as well as by the permalink to the online source of the raw text):

- Friedrich Schlegel, Kritische Fragmente [Lyceum-Fragmente].
Source: Kritische Friedrich-Schlegel-Ausgabe. Erste Abteilung: Kritische Neuausgabe, Band 2, München, Paderborn, Wien, Zürich 1967, S. 147-164.
Erstdruck in: Lyceum der schönen Künste (Berlin), 1. Bd., 2. Teil, 1797.
Permalink: http://www.zeno.org/nid/20005618886

- Friedrich Schelgel, Athenäums-Fragmente [fragments from 1 to 421].
Source: Kritische Friedrich-Schlegel-Ausgabe. Erste Abteilung: Kritische Neuausgabe, Band 2, München, Paderborn, Wien, Zürich 1967, S. 165-256.
Erstdruck in: Athenäum (Berlin), 1. Bd., 2. Stück, 1798.
Permalink: http://www.zeno.org/nid/20005618908

- Novalis, Blüthenstaub.
Source: Novalis: Schriften. Die Werke Friedrich von Hardenbergs. Band 2, Stuttgart 1960–1977, S. 413-464.
Entstanden 1797/98. Erstdruck in: Athenäum (Berlin), 1. Bd., 1. Stück, 1798. Vier Fragmente stammen von Friedrich Schlegel.
Permalink: http://www.zeno.org/nid/20005446929

Each sentence in the treebank file is preceded by some comments introduced by '#', through which the following information is preserved:
- Genre
- Author
- Work
- Number of the fragment. It is based upon the classification adopted in the source raw text. Each time a new fragemnt begins, it is preceded by the comment 'newpar id = [name]', and the number of the fragment is incorporated into the 'sent_id' field as well, followed by the numer of the sentence in that fragment. Moreover, each time a new work begins, it is preceded by the comment '# newdoc id = [name]'. For instance: 

'# newdoc id = bluethenstaub'
'# newpar id = bluethenstaub-f1'
'# author = Novalis'
'# work = Blüthenstaub'
'# sent_id = bluethenstaub-f1-s1'

We made this choice since the treebank is exactly intended as a structured version in dependency formalism of the original texts, therefore we want to preserve the parallelism between the treebanked data and the source texts as much as possible.

# Acknowledgments

Many thanks to Daniel Zeman, who promptly solved some fundamental problems concerning data format, and showed great interest for this project right from the beginning.
...

## References

* (citation)

<pre>
=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: UD v2.4
License: CC BY-NC-SA 4.0
Includes text: yes
Genre: nonfiction
Lemmas: automatic with corrections
UPOS: converted with corrections
XPOS: automatic with corrections
Features: not available
Relations: manual native
Contributors: Salomoni, Alessio
Contributing: elsewhere
Contact: alessio.salomoni@unibg.it
===============================================================================
</pre>
