-- SET search_path TO mimiciii;		-- tried to create everything new in 'public' schema, only old tables remain in 'mimiciii', but sometimes this switch is needed

-- 0) Save all diagnoses in one Field:
		-- takes approx. 45 minutes
		--  all unique admissions are 58.976, total admissions 651047: select count(distinct hadm_id) from mimiciii.diagnoses_icd;
		-- creates new table 'all_diagnoses_icd' in public schema
		-- with new field 'all_icd9_codes'
		select get_all_diagnoses();

-- 1) Cohort: export patient cohort into one .csv (with corresponding Python script)
	-- select * from patient_cohort_view;
	
	-- TODO: Select unique admission with icd_code (Filtering)
	
-- 2) Header: get column names 
	-- SELECT column_name FROM information_schema.columns WHERE table_schema = 'mimiciii' AND table_name = 'view_patient_cohort'
	-- SELECT * FROM information_schema.columns WHERE table_schema = 'mimiciii' AND table_name = 'diagnoses_icd';

-- 3) TODO: get Chartevents per Patient into csv

	
--------- Other Queries ---------	
-- get amount of excluded admissions
	-- select count(hadm_id) from patient_cohort_view where excluded = 0  -- only 13762

-- filter with ICD
 	-- select count(hadm_id) from patient_cohort_with_icd where patient_cohort_with_icd.icd9_code = '42731' -- patient_cohort_with_icd.icd9_code = '7804';
	
-- Checking amount of Diagnoses -> up to 39 diagnoses for one patient -> how should I deal with that??	
-- select max(seq_num) from mimiciii.diagnoses_icd;

-- archive: perform the get-function to get cohort
	-- select * from public.get_patient_cohort();