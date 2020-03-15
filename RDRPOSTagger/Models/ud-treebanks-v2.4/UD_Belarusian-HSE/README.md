# Summary

The Belarusian UD treebank is based on a sample of the news texts included in the Belarusian-Russian parallel subcorpus of the Russian National Corpus,
online search available at: http://ruscorpora.ru/search-para-be.html.



# Tokenization

The low-level tokenization of the Belarusian UD treebank generally adopts the RNC standard.

* In general, tokens are delimited by whitespace. The regexp [А-zА-яЁёУўі\-]+ usually corresponds to one token.
* Punctuation (recognized by the corresponding Unicode property) that is conventionally written adjacent to the preceding or following word is separated during tokenization.
* Each punctuation mark is treated as a single token, e.g. the following sequence: <b>)", -</b> becomes four tokens, <b>)</b> , <b>"</b>, <b>,</b>, and <b>-"</b>. Exceptions are conventional multi-character punctuation marks: <b>--</b> , <b>...</b> , <b>?!</b> ,  etc., and emojis and smileys: <b>:)</b> , <b>^_^</b>, etc.
* Conventional non-cyrillic multi-character terms are tokenized as single tokens: <b>°С</b>, <b>км2</b>.

Some special cases worth mentioning:
* Numerical expressions including decimal numbers, such as <b>245</b>, <b>3,14</b>, are treated as single tokens.
* Time expressions like <b>20:55</b> are splitted into separate tokens (in this case, three { <b>20</b> , <b>:</b> , <b>55</b> }).
* Dates like <b>20.04.2012</b> are splitted into separate tokens (in this case, five { <b>20</b> , <b>.</b> , <b>04</b> , <b>.</b> , <b>2012</b> }).
* Special symbols before and after numerical expressions, as in <b>$500</b> , <b>2,67%</b> , <b>+27°С</b> , are tokenised separately (so, the tokens are { <b>$</b> , <b>500</b> } , { <b>2,67</b> , <b>%</b> } , { <b>+</b> , <b>27</b> , <b>°С</b> }).
* Numerical expressions with hyphen and cyrillic endings (e.g. <b>1-ый</b> “1st”, <b>3-м</b> “3rd.Ins”) as well as adjectives and other non-numerals which contain digits (e.g. <b>79-гадовы</b> “79 year old”, <b>500-годдзе</b> “500th anniversary”) are treated as single tokens.
* Other words with hyphen are treated as single tokens, except for the cases then the first part is inflected. Examples: { <b>з-за</b> } “because of”, { <b>зялёна-шэрых</b> } “green-gray”, { <b>Санкт-Пецярбург</b> } “St. Petersburg”, but { <b>Ростове</b> , <b>-</b> , <b>на</b> , <b>-</b> , <b>Дону</b>} “(in) Rostov on Don”.
* Abbreviations are treated as single tokens, whitespaces split the abbreviations.
* Abbreviations marked by a period, as in <b>стр.</b> “p. (page)”, <b>П.</b> “P. (for Peter)”, are treated as single tokens. If the period overlaps with the end of sentence period then it is written once as a separate token (denoting end-of-sentence), e.g. { <b>1914</b> , <b>г</b> , <b>.</b> } “year 1914”.
* Abbreviations can not contain a period inside, i.e. the patterns like <b>і т.д.</b> “and so on”, <b>да т.п.</b> “and so forth” are splitted into three tokens: { <b>i</b> , <b>т.</b> , <b>д.</b> }, { <b>да</b> , <b>т.</b> , <b>п.</b> }.
* Email addresses, URLs, and tweet-style names are treated as single tokens: {no@mail.ru}, {https://github.com}, {@anna_li}

The Belarusian UD treebank does not contain multiword tokens.



# Morphology

The morphological annotation is adopted from the Russian-Syntagrus UD guidelines and mostly compliant with the RNC morphological standard (exept for "second" cases, comp2, imper2, which were converted to the "primary" tags, and transitivity tags, which were removed).
Lemmas and features were annotated manually.



# Syntax

The data were labeled semi-automatically using the annotation projection from Russian. For that purpose, Russian data were annotated using UDpipe, converted into UD 2.0, and then checked manually.
Belarusian dependency relation tags were checked manually.



# Texts

The source texts are the following:
1) short news articles originally written in Belarusian (and/)or Russian and published by telegraf.by online agency.
Document list: http://search2.ruscorpora.ru/search.xml?env=alpha&text=meta&sort=gr_tagging&lang=ru&doc_g_number_lang=&doc_te_author=&mode=para&doc_te_header=*&author=&doc_g_birthday=&doc_l_birthday=&doc_g_created=&doc_l_created=&doc_te_translator=&doc_lang=bel&doc_lang_trans=rus&doc_g_date_date_trans=&doc_l_date_date_trans=&doc_sphere=%EF%F3%E1%EB%E8%F6%E8%F1%F2%E8%EA%E0
2) short news articles published by http://zviazda.by/.
Document list:
http://zviazda.by/be/news/20170524/1495626198-samaya-prygozhaya-studentka-vuchycca-u-vicebsku
http://zviazda.by/be/news/20160908/1473288543-buduchynya-za-dakladnym-zemlyarobstvam



# Acknowledgments

We thank Uladzimir Koshchanka (Уладзімір Кошчанка, koshul@gmail.com) for providing a part of source texts, Anna Sherbakova (aniezka.sherbakova@gmail.com) for checking the pos and feature labels in two texts.



# Changelog

* 2019-01-05 v2.4
  * Constructions with parataxis, appos, ccomp, xcomp, ccomp, advcl, acl, nmod, passive and depictive constructions manually fixed.
  * UPOS, FEAT manually fixed.
  * Lemmas of PROPN uppercased.
  * New texts (genre: legal nonfiction fiction) added.
* 2018-04-15 v2.2
  * Repository renamed from UD_Belarusian to UD_Belarusian-HSE.
* 2017-11-15 v2.1
  * Flat / appos fixed.
  * New texts added.
* 2017-03-01 v2.0
  * Initial UD release.



<pre>
=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: UD v2.0
License: CC BY-SA 4.0
Includes text: yes
Genre: fiction legal news nonfiction
Lemmas: converted from manual
UPOS: converted from manual
XPOS: manual native
Features: converted from manual
Relations: converted from manual
Contributors: Lyashevskaya, Olga; Peljak-Łapińska, Angelika; Petrova, Daria
Contributing: elsewhere
Contact: olesar@yandex.ru
===============================================================================
</pre>
