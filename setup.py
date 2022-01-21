from setuptools import setup

setup(
    name='crossai',
    version='0.0.1',
    packages=['crossai', 'crossai.ts', 'crossai.ts.ai', 'crossai.ts.models', 'crossai.ts.models.sl',
              'crossai.ts.models.nlp', 'crossai.ts.models.nn1d', 'crossai.ts.models.nn2d', 'crossai.ts.processing',
              'crossai.ts.processing.timeseries', 'crossai.ts.processing.timeseries.motion',
              'crossai.ts.processing.timeseries.signal', 'crossai.ts.processing.timeseries.utilities', 'crossai.nlp',
              'crossai.xai', 'crossai.xai.classification', 'crossai.utilities'],
    url='https://github.com/tzamalisp/cross-ai-lib/',
    license='GNU General Public License v3.0',
    author='Pantelis Tzamalis, Andreas Bardoutsos, Markantonatos Dimitrios',
    author_email='tzamalispantelis@gmail.com, anmp.ce@gmail.com, markantonatosdimitrios@gmail.com',
    description='A library of high-level functionalities capable of building Artificial Intelligence processing '
                'pipelines for Time-Series and Natural Language Processing.'
)

