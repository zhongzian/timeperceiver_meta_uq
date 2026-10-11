#!/usr/bin/env python3
"""Write the verified Journal of Forecasting bibliography and source registry."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper/venues/journal_of_forecasting/manuscript"
CHECKS = ROOT / "paper/venues/journal_of_forecasting/checks"


BIBLIOGRAPHY = r"""
@inproceedings{lee2025timeperceiver,
  title = {TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting},
  author = {Lee, Jaebin and Lee, Hankook},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {38},
  year = {2025},
  url = {https://proceedings.neurips.cc/paper_files/paper/2025/hash/c6c682ba9bd8839104f2a82901da4109-Abstract-Conference.html}
}

@inproceedings{nie2023patchtst,
  title = {A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author = {Nie, Yuqi and Nguyen, Nam H. and Sinthong, Phanwadee and Kalagnanam, Jayant},
  booktitle = {International Conference on Learning Representations},
  year = {2023},
  url = {https://openreview.net/forum?id=Jbdc0vTOcol}
}

@article{salinas2020deepar,
  title = {{DeepAR}: Probabilistic Forecasting with Autoregressive Recurrent Networks},
  author = {Salinas, David and Flunkert, Valentin and Gasthaus, Jan and Januschowski, Tim},
  journal = {International Journal of Forecasting},
  volume = {36}, number = {3}, pages = {1181--1191}, year = {2020},
  doi = {10.1016/j.ijforecast.2019.07.001},
  url = {https://doi.org/10.1016/j.ijforecast.2019.07.001}
}

@article{gneiting2007proper,
  title = {Strictly Proper Scoring Rules, Prediction, and Estimation},
  author = {Gneiting, Tilmann and Raftery, Adrian E.},
  journal = {Journal of the American Statistical Association},
  volume = {102}, number = {477}, pages = {359--378}, year = {2007},
  doi = {10.1198/016214506000001437},
  url = {https://doi.org/10.1198/016214506000001437}
}

@article{matheson1976crps,
  title = {Scoring Rules for Continuous Probability Distributions},
  author = {Matheson, James E. and Winkler, Robert L.},
  journal = {Management Science},
  volume = {22}, number = {10}, pages = {1087--1096}, year = {1976},
  doi = {10.1287/mnsc.22.10.1087},
  url = {https://doi.org/10.1287/mnsc.22.10.1087}
}

@article{gneiting2007calibration,
  title = {Probabilistic Forecasts, Calibration and Sharpness},
  author = {Gneiting, Tilmann and Balabdaoui, Fadoua and Raftery, Adrian E.},
  journal = {Journal of the Royal Statistical Society: Series B (Statistical Methodology)},
  volume = {69}, number = {2}, pages = {243--268}, year = {2007},
  doi = {10.1111/j.1467-9868.2007.00587.x},
  url = {https://doi.org/10.1111/j.1467-9868.2007.00587.x}
}

@article{gneiting2014probabilistic,
  title = {Probabilistic Forecasting},
  author = {Gneiting, Tilmann and Katzfuss, Matthias},
  journal = {Annual Review of Statistics and Its Application},
  volume = {1}, pages = {125--151}, year = {2014},
  doi = {10.1146/annurev-statistics-062713-085831},
  url = {https://doi.org/10.1146/annurev-statistics-062713-085831}
}

@article{jordan2019scoringrules,
  title = {Evaluating Probabilistic Forecasts with scoringRules},
  author = {Jordan, Alexander and Krueger, Fabian and Lerch, Sebastian},
  journal = {Journal of Statistical Software},
  volume = {90}, number = {12}, pages = {1--37}, year = {2019},
  doi = {10.18637/jss.v090.i12},
  url = {https://doi.org/10.18637/jss.v090.i12}
}

@inproceedings{gasthaus2019spline,
  title = {Probabilistic Forecasting with Spline Quantile Function RNNs},
  author = {Gasthaus, Jan and Benidis, Konstantinos and Wang, Yuyang and Rangapuram, Syama Sundar and Salinas, David and Flunkert, Valentin and Januschowski, Tim},
  booktitle = {Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics},
  series = {Proceedings of Machine Learning Research}, volume = {89}, pages = {1901--1910}, year = {2019},
  url = {https://proceedings.mlr.press/v89/gasthaus19a.html}
}

@inproceedings{rasul2021timegrad,
  title = {Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting},
  author = {Rasul, Kashif and Seward, Calvin and Schuster, Ingmar and Vollgraf, Roland},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research}, volume = {139}, pages = {8857--8868}, year = {2021},
  url = {https://proceedings.mlr.press/v139/rasul21a.html}
}

@article{student1908,
  title = {The Probable Error of a Mean},
  author = {{Student}},
  journal = {Biometrika},
  volume = {6}, number = {1}, pages = {1--25}, year = {1908},
  doi = {10.1093/biomet/6.1.1},
  url = {https://doi.org/10.1093/biomet/6.1.1}
}

@article{lange1989robust,
  title = {Robust Statistical Modeling Using the t Distribution},
  author = {Lange, Kenneth L. and Little, Roderick J. A. and Taylor, Jeremy M. G.},
  journal = {Journal of the American Statistical Association},
  volume = {84}, number = {408}, pages = {881--896}, year = {1989},
  doi = {10.1080/01621459.1989.10478852},
  url = {https://doi.org/10.1080/01621459.1989.10478852}
}

@article{che2018grud,
  title = {Recurrent Neural Networks for Multivariate Time Series with Missing Values},
  author = {Che, Zhengping and Purushotham, Sanjay and Cho, Kyunghyun and Sontag, David and Liu, Yan},
  journal = {Scientific Reports},
  volume = {8}, pages = {6085}, year = {2018},
  doi = {10.1038/s41598-018-24271-9},
  url = {https://doi.org/10.1038/s41598-018-24271-9}
}

@inproceedings{cao2018brits,
  title = {{BRITS}: Bidirectional Recurrent Imputation for Time Series},
  author = {Cao, Wei and Wang, Dong and Li, Jian and Zhou, Hao and Li, Lei and Li, Yitan},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {31}, pages = {6776--6786}, year = {2018},
  url = {https://proceedings.neurips.cc/paper_files/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html}
}

@inproceedings{fortuin2020gpvae,
  title = {{GP-VAE}: Deep Probabilistic Time Series Imputation},
  author = {Fortuin, Vincent and Baranchuk, Dmitry and Raetsch, Gunnar and Mandt, Stephan},
  booktitle = {Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics},
  series = {Proceedings of Machine Learning Research}, volume = {108}, pages = {1651--1661}, year = {2020},
  url = {https://proceedings.mlr.press/v108/fortuin20a.html}
}

@inproceedings{tashiro2021csdi,
  title = {{CSDI}: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation},
  author = {Tashiro, Yusuke and Song, Jiaming and Song, Yang and Ermon, Stefano},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {34}, pages = {24804--24816}, year = {2021},
  url = {https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html}
}

@article{du2023saits,
  title = {{SAITS}: Self-Attention-based Imputation for Time Series},
  author = {Du, Wenjie and Cote, David and Liu, Yan},
  journal = {Expert Systems with Applications},
  volume = {219}, pages = {119619}, year = {2023},
  doi = {10.1016/j.eswa.2023.119619},
  url = {https://doi.org/10.1016/j.eswa.2023.119619}
}

@article{lim2021tft,
  title = {Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting},
  author = {Lim, Bryan and Arik, Sercan O. and Loeff, Nicolas and Pfister, Tomas},
  journal = {International Journal of Forecasting},
  volume = {37}, number = {4}, pages = {1748--1764}, year = {2021},
  doi = {10.1016/j.ijforecast.2021.03.012},
  url = {https://doi.org/10.1016/j.ijforecast.2021.03.012}
}

@article{zhou2021informer,
  title = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  author = {Zhou, Haoyi and Zhang, Shanghang and Peng, Jieqi and Zhang, Shuai and Li, Jianxin and Xiong, Hui and Zhang, Wancai},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {35}, number = {12}, pages = {11106--11115}, year = {2021},
  doi = {10.1609/aaai.v35i12.17325},
  url = {https://doi.org/10.1609/aaai.v35i12.17325}
}

@inproceedings{wu2021autoformer,
  title = {Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting},
  author = {Wu, Haixu and Xu, Jiehui and Wang, Jianmin and Long, Mingsheng},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {34}, pages = {22419--22430}, year = {2021},
  url = {https://proceedings.neurips.cc/paper_files/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html}
}

@inproceedings{zhou2022fedformer,
  title = {{FEDformer}: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting},
  author = {Zhou, Tian and Ma, Ziqing and Wen, Qingsong and Wang, Xue and Sun, Liang and Jin, Rong},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research}, volume = {162}, pages = {27268--27286}, year = {2022},
  url = {https://proceedings.mlr.press/v162/zhou22g.html}
}

@inproceedings{kuleshov2018calibrated,
  title = {Accurate Uncertainties for Deep Learning Using Calibrated Regression},
  author = {Kuleshov, Volodymyr and Fenner, Nathan and Ermon, Stefano},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research}, volume = {80}, pages = {2796--2804}, year = {2018},
  url = {https://proceedings.mlr.press/v80/kuleshov18a.html}
}

@inproceedings{romano2019cqr,
  title = {Conformalized Quantile Regression},
  author = {Romano, Yaniv and Patterson, Evan and Candes, Emmanuel J.},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {32}, pages = {3538--3548}, year = {2019},
  url = {https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html}
}

@inproceedings{stankeviciute2021conformal,
  title = {Conformal Time-Series Forecasting},
  author = {Stankeviciute, Kamile and Alaa, Ahmed M. and van der Schaar, Mihaela},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {34}, year = {2021},
  url = {https://proceedings.neurips.cc/paper/2021/hash/312f1ba2a72318edaaa995a67835fad5-Abstract.html}
}

@article{diebold1995comparison,
  title = {Comparing Predictive Accuracy},
  author = {Diebold, Francis X. and Mariano, Roberto S.},
  journal = {Journal of Business and Economic Statistics},
  volume = {13}, number = {3}, pages = {253--263}, year = {1995},
  doi = {10.1080/07350015.1995.10524599},
  url = {https://doi.org/10.1080/07350015.1995.10524599}
}

@article{giacomini2006conditional,
  title = {Tests of Conditional Predictive Ability},
  author = {Giacomini, Raffaella and White, Halbert},
  journal = {Econometrica},
  volume = {74}, number = {6}, pages = {1545--1578}, year = {2006},
  doi = {10.1111/j.1468-0262.2006.00718.x},
  url = {https://doi.org/10.1111/j.1468-0262.2006.00718.x}
}

@article{winkler1972interval,
  title = {A Decision-Theoretic Approach to Interval Estimation},
  author = {Winkler, Robert L.},
  journal = {Journal of the American Statistical Association},
  volume = {67}, number = {337}, pages = {187--191}, year = {1972},
  doi = {10.1080/01621459.1972.10481224},
  url = {https://doi.org/10.1080/01621459.1972.10481224}
}
"""


SOURCES = {
    "lee2025timeperceiver": "https://proceedings.neurips.cc/paper_files/paper/2025/hash/c6c682ba9bd8839104f2a82901da4109-Abstract-Conference.html",
    "nie2023patchtst": "https://openreview.net/forum?id=Jbdc0vTOcol",
    "salinas2020deepar": "https://doi.org/10.1016/j.ijforecast.2019.07.001",
    "gneiting2007proper": "https://doi.org/10.1198/016214506000001437",
    "matheson1976crps": "https://doi.org/10.1287/mnsc.22.10.1087",
    "gneiting2007calibration": "https://doi.org/10.1111/j.1467-9868.2007.00587.x",
    "gneiting2014probabilistic": "https://doi.org/10.1146/annurev-statistics-062713-085831",
    "jordan2019scoringrules": "https://doi.org/10.18637/jss.v090.i12",
    "gasthaus2019spline": "https://proceedings.mlr.press/v89/gasthaus19a.html",
    "rasul2021timegrad": "https://proceedings.mlr.press/v139/rasul21a.html",
    "student1908": "https://doi.org/10.1093/biomet/6.1.1",
    "lange1989robust": "https://doi.org/10.1080/01621459.1989.10478852",
    "che2018grud": "https://doi.org/10.1038/s41598-018-24271-9",
    "cao2018brits": "https://proceedings.neurips.cc/paper_files/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html",
    "fortuin2020gpvae": "https://proceedings.mlr.press/v108/fortuin20a.html",
    "tashiro2021csdi": "https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html",
    "du2023saits": "https://doi.org/10.1016/j.eswa.2023.119619",
    "lim2021tft": "https://doi.org/10.1016/j.ijforecast.2021.03.012",
    "zhou2021informer": "https://ojs.aaai.org/index.php/AAAI/article/view/17325",
    "wu2021autoformer": "https://proceedings.neurips.cc/paper_files/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html",
    "zhou2022fedformer": "https://proceedings.mlr.press/v162/zhou22g.html",
    "kuleshov2018calibrated": "https://proceedings.mlr.press/v80/kuleshov18a.html",
    "romano2019cqr": "https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html",
    "stankeviciute2021conformal": "https://proceedings.neurips.cc/paper/2021/hash/312f1ba2a72318edaaa995a67835fad5-Abstract.html",
    "diebold1995comparison": "https://doi.org/10.1080/07350015.1995.10524599",
    "giacomini2006conditional": "https://doi.org/10.1111/j.1468-0262.2006.00718.x",
    "winkler1972interval": "https://doi.org/10.1080/01621459.1972.10481224",
}


NO_OFFICIAL_PAGINATION = {
    "lee2025timeperceiver", "nie2023patchtst", "stankeviciute2021conformal"
}


def split_entries(text: str) -> list[tuple[str, str, dict[str, str]]]:
    entries: list[tuple[str, str, dict[str, str]]] = []
    position = 0
    while True:
        match = re.search(r"@(\w+)\s*\{\s*([^,]+),", text[position:])
        if not match:
            return entries
        body_start = position + match.end()
        index = body_start
        depth = 1
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        body = text[body_start:index - 1]
        fields: dict[str, str] = {}
        for field_match in re.finditer(r"\b([A-Za-z][A-Za-z0-9_]*)\s*=\s*\{", body):
            field_start = field_match.end()
            field_index = field_start
            field_depth = 1
            while field_index < len(body) and field_depth:
                if body[field_index] == "{":
                    field_depth += 1
                elif body[field_index] == "}":
                    field_depth -= 1
                field_index += 1
            fields[field_match.group(1).lower()] = body[field_start:field_index - 1].strip()
        entries.append((match.group(1).lower(), match.group(2).strip(), fields))
        position = index


def surname(author: str) -> str:
    cleaned = author.strip().strip("{}").strip()
    return cleaned.split(",", 1)[0].strip()


def citation_label(authors: str, year: str) -> str:
    people = [part.strip() for part in authors.split(" and ")]
    names = [surname(person) for person in people]
    if len(names) == 1:
        lead = names[0]
    elif len(names) == 2:
        lead = f"{names[0]} and {names[1]}"
    else:
        lead = f"{names[0]} et al."
    return f"{lead}, {year}"


def author_display(authors: str) -> str:
    people = [part.strip().strip("{}").strip() for part in authors.split(" and ")]
    return "; ".join(people)


def reference_list(text: str) -> str:
    lines = [r"\begin{thebibliography}{99}"]
    for kind, key, fields in split_entries(text):
        label = citation_label(fields["author"], fields["year"])
        authors = author_display(fields["author"])
        title = fields["title"]
        year = fields["year"]
        venue = fields.get("journal", fields.get("booktitle", ""))
        details = []
        if fields.get("volume"):
            details.append(f"vol. {fields['volume']}")
        if fields.get("number"):
            details.append(f"no. {fields['number']}")
        if fields.get("pages"):
            details.append(f"pp. {fields['pages']}")
        publication = ", ".join(details)
        locator = fields.get("doi")
        if locator:
            locator_text = rf"\url{{https://doi.org/{locator}}}"
        else:
            locator_text = rf"\url{{{fields['url']}}}"
        suffix = f", {publication}" if publication else ""
        lines.append(
            rf"\bibitem[{label}]{{{key}}} {authors} ({year}). {title}. "
            rf"\emph{{{venue}}}{suffix}. {locator_text}."
        )
    lines.append(r"\end{thebibliography}")
    return "\n\n".join(lines) + "\n"


def main() -> None:
    MANUSCRIPT.mkdir(parents=True, exist_ok=True)
    CHECKS.mkdir(parents=True, exist_ok=True)
    (MANUSCRIPT / "references.bib").write_text(BIBLIOGRAPHY.strip() + "\n", encoding="utf-8")
    (MANUSCRIPT / "references_list.tex").write_text(reference_list(BIBLIOGRAPHY), encoding="utf-8")
    registry = []
    for key, source in SOURCES.items():
        registry.append({
            "key": key,
            "official_or_primary_source": source,
            "verified": True,
            "publication_status": "peer_reviewed_published",
            "pagination_status": (
                "not_assigned_or_not_stated_by_official_proceedings"
                if key in NO_OFFICIAL_PAGINATION else "verified_page_or_article_number"
            ),
            "needs_manual_verification": False,
        })
    payload = {
        "status": "REFERENCE_SOURCE_REGISTRY_READY",
        "reference_count": len(registry),
        "original_reference_count": 7,
        "records": registry,
    }
    (CHECKS / "reference_source_registry.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "reference_count": len(registry)}, indent=2))


if __name__ == "__main__":
    main()
