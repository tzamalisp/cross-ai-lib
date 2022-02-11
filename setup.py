from setuptools import setup

setup(
    name='crossai',
    version='0.0.0.3',
    packages=['crossai', 'crossai.ts', 'crossai.ts.processing',
              'crossai.ts.processing.motion', 'crossai.ts.processing.signal',
              'crossai.ts.processing.utilities', 'crossai.ts.predictions',
              'crossai.nlp', 'crossai.nlp.text_processing', 'crossai.xai',
              'crossai.xai.classification', 'crossai.models',
              'crossai.models.nn1d', 'crossai.utilities'],
    url='https://github.com/tzamalisp/cross-ai-lib/',
    license='GNU General Public License v3.0',
    author='Pantelis Tzamalis, Andreas Bardoutsos, Markantonatos Dimitrios',
    author_email='tzamalispantelis@gmail.com, anmp.ce@gmail.com, markantonatosdimitrios@gmail.com',
    description='A library of high-level functionalities capable of building Artificial Intelligence processing '
                'pipelines for Time-Series and Natural Language Processing.',
    install_requires=["alibi==0.6.4", "auto_mix_prep==0.2.0",
                      "lexicalrichness==0.1.4",
                      "matplotlib==3.5.1",
                      "nltk==3.7",
                      "numpy==1.19.5",
                      "pandas==1.4.0",
                      "pymongo==4.0.1",
                      "scipy==1.8.0",
                      "spacy==3.2.2", "spacytextblob==3.0.1",
                      "tensorflow==2.7.0", "tensorflow_addons==0.15.0",
                      "tqdm==4.62.3", "transforms3d==0.3.1", "tsaug==0.2.1",
                      "wordcloud==1.8.1"]
)
