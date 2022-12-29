-- This file creates a table icd9_code | short_title | long_title | source for all chosen icd9_codes (diagnosis and procedures) for this analysis. 
-- The table will be used to connect a icd9_code to its title
-- There are 18.449 available codes

SELECT 
	icd9_code, 
	short_title, 
	long_title, 
	'diagnose' as source
FROM mimiciii.d_icd_diagnoses
UNION ALL SELECT 
	icd9_code, 
	short_title, 
	long_title, 
	'procedure' as source
FROM mimiciii.d_icd_procedures
ORDER BY icd9_code;
