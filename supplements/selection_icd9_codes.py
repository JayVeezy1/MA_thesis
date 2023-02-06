# This file can be used to manually select the use-case, more precise all relevant icd9-codes.

# Complete List of icd9-codes: https://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes/codes


# Myocardial Infarct related codes

# 410.00-411.1 -> STEMI (ST-elevation myocardial infarction)
STEMI_codes = [41000, 41001, 41002, 41010, 41011, 41012, 41020, 41021, 41022, 41030, 41031,
               41032, 41040, 41041, 41042, 41050, 41051, 41052, 41060, 41061, 41062, 41080,
               41081, 41082, 41090, 41091, 41092, 4110, 4111]

# 410.70 - 410.72 -> NSTEMI (non-ST-elevation myocardial infarction)
NSTEMI_codes = [41070, 41071, 41072]

selected_myocardial_infarct_codes = [41000, 41001, 41002, 41010, 41011, 41012, 41020, 41021, 41022, 41030, 41031,
                                     41032, 41040, 41041, 41042, 41050, 41051, 41052, 41060, 41061, 41062, 41080,
                                     41081, 41082, 41090, 41091, 41092, 4110, 4111, 41070, 41071, 41072]



# Stroke related codes
icd9_00_cerebrovascular_general: list = [430, 431, 432, 4329, 433, 4330, 4331, 4332, 434, 4340, 43400, 43401, 4341,
                                         43410, 43411,
                                         435, 4350, 4351, 4352, 4353, 4359, 436, 437, 4370, 4371, 4372, 4373, 4374,
                                         4375, 4376,
                                         4377, 438, 4380, 4381, 43810, 43811, 43812, 43819, 4382, 43820, 43821, 43822,
                                         4383,
                                         4384, 4385, 4388, 43881, 43882, 43883, 43884, 43885, 4389]
# stroke-type reference: https://health.mo.gov/data/mica/CDP_MICA/StrokeDefofInd.html
# hemorrhage: 430, 431, 432, 4329,
# ischemic (+TIA): 433, 4330, 4331, 4332, 434, 4340, 43400, 43401, 4341, 43411, 435, 4350, 4351, 4353, 4359, 436
# other (+late_effects_of_stroke): 437, 4370, 4371, 4372, 4373, 4374, 438, 4381, 43811, 4382, 43820, 4383, 4384, 4385, 4388, 43882, 43885
selected_stroke_codes: list = [430, 431, 432, 4329, 433, 4330, 4331, 4332, 434, 4340, 43400, 43401, 4341, 43411,
                                 435, 4350, 4351, 4353, 4359, 436, 437, 4370, 4371, 4372, 4373, 4374,
                                 438, 4381, 43811, 4382, 43820, 4383, 4384, 4385, 4388, 43882, 43885]

# while we filter for all the icd9_codes listed above, the actual distribution of cases is:
"""
icd9_code   occurrence
431     	535
43411       275
430	        259 
4373    	88
43820   	87
4370    	58
43811   	41
43401   	33
4329	    26
43882   	16
4359	    16
4372	    11
4371	    3
4374	    3

Total       1451
"""

# Overview of cerebrovascular related ICD9-Categories:
"""
Cerebrovascular disease (430–438)
430        Subarachnoid hemorrhage
431        Intracerebral hemorrhage
432        Other and unspecified intracranial hemorrhage
4329       Hemorrhage, intracranial, NOS
433        Occlusion and stenosis of precerebral arteries
4330       Occlusion and stenosis of basilar artery
4331       Occlusion and stenosis of carotid artery
4332       Occlusion and stenosis of vertebral artery
434        Occlusion of cerebral arteries
4340       Cerebral thrombosis
43400      Cerebral thrombosis without cerebral infarction
43401      Cerebral thrombosis with cerebral infarction
4341       Cerebral embolism
43410      Cerebral embolism without cerebral infarction
43411      Cerebral embolism with cerebral infarction
435        Transient cerebral ischemia
4350       Basilar artery syndrome
4351       Vertebral artery syndrome
4352       Subclavian steal syndrome
4353       Vertebrobasilar artery syndrome
4359       Transient ischemic attack, unspec.
436        Acute but ill-defined cerebrovascular disease
437        Other and ill-defined cerebrovascular disease
4370       Cerebral atherosclerosis
4371       Other generalized ischemic cerebrovascular disease
4372       Hypertensive encephalopathy
4373       Cerebral aneurysm nonruptured
4374       Cerebral arteritis
4375       Moyamoya disease
4376       Nonpyogenic thrombosis of intracranial venous sinus
4377       Transient global amnesia
438        Late effects of cerebrovascular disease
4380       Cognitive deficits
4381       Speech and language deficits
43810      Speech and language deficits, unspecified
43811      Aphasia
43812      Dysphasia
43819      Other speech and language deficits
4382       Hemiplegia/hemiparesis
43820      Hemiplegia affecting unspecified side
43821      Hemiplegia affecting dominant side
43822      Hemiplegia affecting nondominant side
4383       Monoplegia of upper limb
4384       Monoplegia of lower limb
4385       Other paralytic syndrome
4388       Other late effects of cerebrovascular disease
43881      Apraxia cerebrovascular disease
43882      Dysphagia cerebrovascular disease
43883      Facial weakness
43884      Ataxia
43885      Vertigo
4389       CVA, late effect, unspec.
"""

# Overview of all general ICD-9 Categories (https://en.wikipedia.org/wiki/List_of_ICD-9_codes)
"""
List of ICD-9 codes 001–139: infectious and parasitic diseases
List of ICD-9 codes 140–239: neoplasms
List of ICD-9 codes 240–279: endocrine, nutritional and metabolic diseases, and immunity disorders
List of ICD-9 codes 280–289: diseases of the blood and blood-forming organs
List of ICD-9 codes 290–319: mental disorders
List of ICD-9 codes 320–389: diseases of the nervous system and sense organs
List of ICD-9 codes 390–459: diseases of the circulatory system
List of ICD-9 codes 460–519: diseases of the respiratory system
List of ICD-9 codes 520–579: diseases of the digestive system
List of ICD-9 codes 580–629: diseases of the genitourinary system
List of ICD-9 codes 630–679: complications of pregnancy, childbirth, and the puerperium
List of ICD-9 codes 680–709: diseases of the skin and subcutaneous tissue
List of ICD-9 codes 710–739: diseases of the musculoskeletal system and connective tissue
List of ICD-9 codes 740–759: congenital anomalies
List of ICD-9 codes 760–779: certain conditions originating in the perinatal period
List of ICD-9 codes 780–799: symptoms, signs, and ill-defined conditions
List of ICD-9 codes 800–999: injury and poisoning
List of ICD-9 codes E and V codes: external causes of injury and supplemental classification
"""
icd9_01_infections: list = ['001-139']
icd9_02_neoplasm: list = ['140–239']
icd9_03_metabolic: list = ['240–279']
icd9_04_blood: list = ['280–289']
icd9_05_mental_disorders: list = ['290–319']
icd9_06_nervous_system: list = ['320–389']
icd9_07_circulatory_system: list = ['390–459']  # stroke relevant codes further below
icd9_08_respiratory_system: list = ['460–519']
icd9_09_digestive_system: list = ['520–579']
icd9_10_genitourinary_system: list = ['580–629']
icd9_11_pregnancy: list = ['630–679']
icd9_12_skin: list = ['680–709']
icd9_13_musculoskeletal: list = ['710–739']
icd9_14_congenital_anomalies: list = ['740–759']
icd9_15_perinatal_period: list = ['760–779']
icd9_16_ill_defined: list = ['780–799']
icd9_17_injury_poisoning: list = ['800–999']
icd9_18_E_and_V_external_supplemental: list = ['E and V codes: external causes of injury and supplemental']
