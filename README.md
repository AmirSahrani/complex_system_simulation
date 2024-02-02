<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="Complex System Simulation"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![MIT License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/amirsahrani/complex_system_simulation">
    <img src="figures/round_spiral_round.gif" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Criticality in the Cortex</h3>

  <p align="center">
  Our project investigating criticality in the cortex through a Cellular Automata model and a branching model
  </p>

</div>





<!-- GETTING STARTED -->
## Getting Started

To jump straight into the figures and results you can take a look at the `analysis.ipynb` notebook. If you would like to look into the code behind these results, the cellular automata model can be found in the `sandpile.py` file (the name is a bit misleading, but this is model is a bit like a cross over between a CA and a sandpile model). The branching model can be found in the `branching.py` file. All python files  in the `utils` folder contain code used to deal with data, plotting or general utilies, these files are use to make the notebook more concise and reader friendly.

If you would like to browser the docs, you can run the following command in the terminal once you have clone this repo.
```
pydoc -b
```
Most large functions or classes have documentation explaining there attributes and methods.

A few essential functions have tests in the test.py file, these can be run using the following:
```
pytest test.py
```
All tests should pass!

### Prerequisites

A `requirements.txt` file has been provided, this should install any necessary libraries!



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/amirsahrani/complex_system_simulation/graphs/contributors
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/amirsahrani/complex_system_simulation/LICENSE.txt