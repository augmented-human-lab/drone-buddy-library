Google Text Recognition Module Installation
========================


The official documentation can be found `here <https://cloud.google.com/use-cases/ocr??utm_source=google&utm_source=google&utm_medium=cpc&utm_campaign=japac-SG-all-en-dr-SKWS-all-all-trial-DSA-dr-1605216&utm_content=text-ad-none-none-DEV_c-CRE_647923039857-ADGP_Hybrid+%7C+SKWS+-+BRO+%7C+DSA+~+All+Webpages-KWID_39700075148142355-aud-1596662388934:dsa-19959388920&userloc_9062542-network_g&utm_term=KW_&gad_source=1&gclid=Cj0KCQiAn-2tBhDVARIsAGmStVkoIwR9T1wHOGz7mvZOl4096RF4VVmipdEqAl2g9knnke5Wv0qhR0UaAnYdEALw_wcB&gclsrc=aw.ds&hl=en>`_

Installation
-------------

To install pyttsx3 Integration, run the following snippet, which will install the required dependencies

.. code-block::

    pip install dronebuddylib[TEXT_RECOGNITION]



Usage
-------------


.. code-block:: python

     engine_configs = EngineConfigurations({})
     engine = TextRecognitionEngine(TextRecognitionAlgorithm.GOOGLE_VISION, engine_configs)
     image_path = r'C:\Users\Public\projects\drone-buddy-library\test\test_images\test_image_clear.jpg'
     result = engine.recognize_text(image_path)



Output
------

.. code-block:: json
    {
            'text': "",
            'locale': "",
            'full_information': ""
    }
