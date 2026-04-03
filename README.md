<h2>Text comparison tool</h2>
This tool is a web interface allowing to compare lexically and semantically two texts.<br>
It pairs similar sentences based on the selected metrics. <br><br>

This tool allows you to upload large text files (hundreds of pages) and find pairs of sentences in reasonable time (few minutes at most).<br>

<h2>Instructions to install</h2>

1. Make sure a recent version of Python is installed on your computer.<br>
   `python --version`

2. Download this repo (only the first time):
   `git clone https://github.com/obtic-sorbonne/comparaison.git`

3. (Optional) Create a virtual environment:
    `python3 -m venv venv`
    `source venv/bin/activate`  # On Windows: `venv\Scripts\activate`
   
3. Install libraries by typing in the terminal (only the first time, in the same installed folder):
   `pip install -r requirements.txt`<br>
 
5. In terminal, type `python main.py` <br>

6. Open `http://127.0.0.1:5000` in the web browser.

<h2>Authors</h2>
Made during a 2-month summer internship (2024) in the ObTIC Sorbonne Universite team.<br>
Author: Clément MARIE <br>
Supervised by: Motasem Alrahabi <br>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
