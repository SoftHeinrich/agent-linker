# API records added to the standalone RW draft

Command: `python3 research/fse-2027-reviewer-audit/expand_rw_draft.py`

Each row was fetched independently. Crossref bibliography fields come from its BibTeX transform; when the transform year differs from Crossref's print/published year, the latter API field supplies the draft year. Colliding or year-mismatched local keys are renamed. arXiv entries use only Atom fields. The search sheet supplied locator and evidence ID only.

| ID | API metadata | Title | Authors | Year | Key |
|---|---|---|---|---:|---|
| E007 | [record](https://api.crossref.org/works/10.1002%2Fsmr.419) | Viability for codifying and documenting architectural design decisions with tool support | Rafael Capilla, Juan C. Dueñas, Francisco Nava | 2010 | `RW_E007` |
| E009 | [record](https://api.crossref.org/works/10.1016%2Fj.jss.2015.08.054) | 10 years of software architecture knowledge management: Practice and future | Rafael Capilla, Anton Jansen, Antony Tang, Paris Avgeriou, Muhammad Ali Babar | 2016 | `Capilla_2016` |
| E010 | [record](https://export.arxiv.org/api/query?id_list=1802.04015) | Toward Architectural Knowledge Sustainability. New Opportunities to Extend the Longevity of Systems | Rafael Capilla, Elisa Yumi Nakagawa, Uwe Zdun, Carlos Carrillo | 2018 | `RW_E010` |
| E022 | [record](https://export.arxiv.org/api/query?id_list=2401.01508) | Practical Guidelines for the Selection and Evaluation of Natural Language Processing Techniques in Requirements Engineering | Mehrdad Sabetzadeh, Chetan Arora | 2024 | `RW_E022` |
| E030 | [record](https://export.arxiv.org/api/query?id_list=2308.03784) | Improving Requirements Completeness: Automated Assistance through Large Language Models | Dipeeka Luitel, Shabnam Hassani, Mehrdad Sabetzadeh | 2023 | `RW_E030` |
| E031 | [record](https://export.arxiv.org/api/query?id_list=2302.04792) | Using Language Models for Enhancing the Completeness of Natural-language Requirements | Dipeeka Luitel, Shabnam Hassani, Mehrdad Sabetzadeh | 2023 | `RW_E031` |
| E039 | [record](https://api.crossref.org/works/10.1016%2Fj.infsof.2013.08.004) | Enhancing software artefact traceability recovery processes with link count information | Gabriele Bavota, Andrea De Lucia, Rocco Oliveto, Genoveffa Tortora | 2014 | `Bavota_2014` |
| E040 | [record](https://api.crossref.org/works/10.1016%2Fj.jss.2013.10.019) | Recovering test-to-code traceability using slicing and textual analysis | Abdallah Qusef, Gabriele Bavota, Rocco Oliveto, Andrea De Lucia, Dave Binkley | 2014 | `Qusef_2014` |
| E041 | [record](https://api.crossref.org/works/10.1002%2Fsmr.1573) | Evaluating test‐to‐code traceability recovery methods through controlled experiments | Abdallah Qusef, Gabriele Bavota, Rocco Oliveto, Andrea De Lucia, David Binkley | 2013 | `RW_E041` |
| E059 | [record](https://api.crossref.org/works/10.1007%2Fs10664-023-10397-6) | Detecting outdated code element references in software repository documentation | Wen Siang Tan, Markus Wagner, Christoph Treude | 2024 | `RW_E059` |
| E060 | [record](https://api.crossref.org/works/10.1007%2Fs10664-023-10325-8) | 18 million links in commit messages: purpose, evolution, and decay | Tao Xiao, Sebastian Baltes, Hideaki Hata, Christoph Treude, Raula Gaikovina Kula, Takashi Ishio, Kenichi Matsumoto | 2023 | `Xiao_2023` |
| E062 | [record](https://api.crossref.org/works/10.1145%2F3643773) | Generative AI for Pull Request Descriptions: Adoption, Impact, and Developer Interventions | Tao Xiao, Hideaki Hata, Christoph Treude, Kenichi Matsumoto | 2024 | `Xiao_2024` |
| E063 | [record](https://api.crossref.org/works/10.1007%2Fs10515-023-00407-8) | Large language models for qualitative research in software engineering: exploring opportunities and challenges | Muneera Bano, Rashina Hoda, Didar Zowghi, Christoph Treude | 2024 | `RW_E063` |
| E065 | [record](https://api.crossref.org/works/10.1109%2FTSE.2023.3348172) | Code Review Automation: Strengths and Weaknesses of the State of the Art | Rosalia Tufano, Ozren Dabić, Antonio Mastropaolo, Matteo Ciniselli, Gabriele Bavota | 2024 | `RW_E065` |

Result: 14 API records fetched; 10 Crossref and 4 arXiv; all BibTeX keys unique.

Crossref transform-year differences (online vs. print/published): E007 2009→2010, E041 2012→2013, E059 2023→2024, E063 2023→2024.
