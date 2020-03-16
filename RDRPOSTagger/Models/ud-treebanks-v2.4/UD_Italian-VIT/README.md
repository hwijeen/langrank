# Summary

The UD_Italian-VIT corpus was obtained by conversion from VIT (Venice Italian Treebank), developed at the Laboratory of Computational Linguistics of the Università Ca' Foscari in Venice (Delmonte et al. 2007; Delmonte 2009; http://rondelmo.it/resource/VIT/Browser-VIT/index.htm).

# Introduction

The VIT is the effort of the collaboration of people working at the Laboratory of Computational Linguistics (LCL) of the University of Venice in the years 1995-2005. It is partly the result of annotation carried out and partly related to the development of a lexicon, a morphological analyzer, a tagger, a deep parser of Italian. All these resources were finally ready at the beginning of the 90s when the LCL got involved in the first Italian national projects. 

VIT originated as a constituency based treebank following the theoretical framework described in (Delmonte et al. 2007), and was later converted into a dependency representation in ConLL-X format (Delmonte 2009). The annotation follows general X-bar criteria with 29 constituency labels and 102 PoS tags. VIT is also made available in a broad annotation version with 10 constituency labels and 22 PoS tags for machine learning purposes. The format is plain text with square bracketing and a UPenn style version which is readable by the open source query language has been also provided. The VIT contains about 272,000 words distributed over six different domains, and this is what makes it so relevant for the study of the structure of Italian language. VIT includes linguistic materials of diverse nature, extracted from five different text genres: news (170,000 words), burocratic (20,000 words), political (40,000 words), Economic and financial (12,000 words), scientific (20,000 words) and literary (10,000 words) genres (Delmonte et al. 2007). In addition, some 60,000 tokens of spoken dialogues in different Italian varieties were annotated.

Similarly to what we did for other Italian treebanks, the UD version of the VIT treebank was obtained by first converting to an unriched version of the MIDT (Merged Italian Dependency Treebank) scheme (Bosco, Montemagni, Simi 2012). Then a further conversion step from MIDT+ to UDv2 was performed. Conversion was followed by a series of semi-automatic harmonization steps, in order to compensate for several differences in the use of the target annotation scheme with respect to the other Italian treebanks.
The splitting into training, devel and test was done maintaining as much as possible the original sequence and respecting the proportions indicated in the guidelines (80%, 10%, 10%).

# Acknowledgments

We are indebted to Rodolfo Del Monte and his collaborators, Antonella Bristot and Sara Tonelli, for the initial work on the VIT treebank; we also acknowledge the contribution of Linda Alfieri and Elzara Khaialieva to the implementation of the conversion process from VIT to MIDT+, which consisted in setting up the automatic conversion rules and in checking the treebank manually.

## References

 * Alfieri L., Tamburini F. (2016). (Almost) Automatic Conversion of the 
 Venice Italian Treebank into the Merged Italian Dependency Treebank 
 Format. In Proc. of the Third Italian Conference on Computational 
 Linguistics - CLiC-IT 2016, Napoli, 5-6 December 2016, 19-23.
 
* Bosco C., Montemagni S., Simi, M. (2012) Harmonization and Merging of two Italian Dependency Treebanks, Workshop on Merging of Language Resources, in Proceedings of LREC 2012, Workshop on Language Resource Merging, Instanbul, May 2012, ELRA, pp. 23-30.
 
* Rodolfo Delmonte, Antonella Bristot, and Sara Tonelli (2007). VIT - 
 Venice Italian Treebank: Syntactic and Quantitative Features. In Proc. 
 Sixth International Workshop on Treebanks and Linguistic Theories.
 
* Delmonte, R. (2009). Treebanking in VIT: from Phrase Structure to 
 Dependency Representation. In Sergei Nirenburg (ed.) Language 
 Engineering for Lesser-Studied Languages, pages 51–81. IOS Press, 
 Amsterdam, The Netherlands.
 
* Tamburini F. (2017). Semgrex-Plus: a Tool for Automatic 
 Dependency-Graph Rewriting In Proc. of the Fourth International 
 Conference on Dependency Linguistics - Depling 2017, Pisa, 18-20 
 September 2017, 248-254.

<pre>
=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: UD v2.4
License: CC BY-NC-SA 3.0
Includes text: yes
Genre: nonfiction news
Lemmas: converted from manual
UPOS: converted from manual
XPOS: manual native
Features: converted from manual
Relations: converted from manual
Contributors: Tamburini, Fabio; Simi, Maria; Bosco, Cristina
Contributing: elsewhere
Contact: simi@di.unipi.it
===============================================================================
</pre>
