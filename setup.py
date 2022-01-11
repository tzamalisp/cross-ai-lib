from setuptools import setup

setup(
    name='cross-ai-lib',
    version='0.0.1',
    packages=['crossai', 'crossai.ai', 'crossai.models', 'crossai.models.sl',
              'crossai.models.nlp', 'crossai.models.nn1d',
              'crossai.models.nn2d', 'crossai.xplain',
              'crossai.xplain.classification', 'crossai.utilities',
              'crossai.processing', 'crossai.processing.nlp',
              'crossai.processing.timeseries',
              'crossai.processing.timeseries.motion',
              'crossai.processing.timeseries.signal',
              'crossai.processing.timeseries.utilities'],
    url='https://github.com/tzamalisp/cross-ai-lib/',
    license='GNU General Public License v3.0',
    author='Pantelis Tzamalis, Andreas Bardoutsos, Markantonatos Dimitrios',
    author_email='tzamalispantelis@gmail.com, anmp.ce@gmail.com, markantonatosdimitrios@gmail.com',
    description='A library of high-level functionalities capable of building Artificial Intelligence processing pipelines for Time-Series and Natural Language Processing.'
)
