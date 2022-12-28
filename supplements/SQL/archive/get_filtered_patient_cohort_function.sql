create or replace function get_filtered_patient_cohort(icd9_selected_list varchar[])		
returns table (
		hadm_id						integer,
		icustay_id					integer,
		intime						timestamp without time zone,
		outtime						timestamp without time zone,
		los 						double precision,
		icustays_count				bigint,
		age 						numeric,
		patientweight				numeric,
		dob 						date,
		dod 						date,
		death_in_hosp 				int,
		death_3_days 				int,
		death_30_days 				int,
		death_365_days 				int,				
		gender						varchar(5),
		ethnicity					varchar(200),
		first_service				varchar(20),
		dbsource					varchar(20),
		subject_id					int,
		seq_num						int,
		icd9_code					varchar(10),
		all_icd9_codes 				varchar[]
) 
language plpgsql
as $body$
declare
-- variable declaration with ;	
v_selected_code varchar(10);
	
begin
	-- 0. step create list from input string
	if (SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename  = 'temp_icd9_codes_selected')) then	
		DROP TABLE temp_icd9_codes_selected CASCADE;		-- CASCADE needed because view patient_cohort_filtered is related
	end if;
	
	CREATE TABLE temp_icd9_codes_selected(icd9_codes varchar(10));
	
	FOREACH v_selected_code IN ARRAY icd9_selected_list LOOP
		INSERT INTO temp_icd9_codes_selected (icd9_codes)
		VALUES (v_selected_code);
	END LOOP;

	-- not ideal code because if clause very long, but it works
	IF (SELECT COUNT(*) FROM temp_icd9_codes_selected) = 0 THEN
		create or replace view patient_cohort_filtered
		AS		
		-- 1. step get all patients (no filter)
		with cohort as
		( 
		SELECT
			patient_cohort_with_icd.hadm_id,			
			patient_cohort_with_icd.icustay_id,		
			patient_cohort_with_icd.intime,			
			patient_cohort_with_icd.outtime,			
			patient_cohort_with_icd.los,
			patient_cohort_with_icd.icustays_count,
			patient_cohort_with_icd.age,
			patient_cohort_with_icd.patientweight,
			patient_cohort_with_icd.dob,
			patient_cohort_with_icd.dod,
			patient_cohort_with_icd.death_in_hosp,
			patient_cohort_with_icd.death_3_days,
			patient_cohort_with_icd.death_30_days,
			patient_cohort_with_icd.death_365_days,			
			patient_cohort_with_icd.gender,			
			patient_cohort_with_icd.ethnicity,		
			patient_cohort_with_icd.first_service,	
			patient_cohort_with_icd.dbsource,		
			patient_cohort_with_icd.subject_id,		
			patient_cohort_with_icd.seq_num,			
			patient_cohort_with_icd.icd9_code,		
			patient_cohort_with_icd.all_icd9_codes,
			ROW_NUMBER() OVER (PARTITION BY 
								patient_cohort_with_icd.hadm_id, 
								patient_cohort_with_icd.icustay_id, 
								patient_cohort_with_icd.subject_id
								ORDER BY 
								patient_cohort_with_icd.hadm_id, 
								patient_cohort_with_icd.icustay_id, 
								patient_cohort_with_icd.subject_id,
							   	patient_cohort_with_icd.seq_num
			) rownum
		FROM patient_cohort_with_icd 				 -- This view must be created before running this function here. 
		-- no Filtering for ICD9 Code if no code in parameter list
		ORDER BY patient_cohort_with_icd.hadm_id, patient_cohort_with_icd.seq_num
		)
		
		SELECT 
			cohort.hadm_id,			
			cohort.icustay_id,		
			cohort.intime,			
			cohort.outtime,			
			cohort.los,
			cohort.icustays_count,
			cohort.age,
			cohort.patientweight,
			cohort.dob,
			cohort.dod,
			cohort.death_in_hosp,
			cohort.death_3_days,
			cohort.death_30_days,
			cohort.death_365_days,				
			cohort.gender,			
			cohort.ethnicity,		
			cohort.first_service,	
			cohort.dbsource,		
			cohort.subject_id,		
			cohort.seq_num,
			cohort.icd9_code,		
			cohort.all_icd9_codes
		FROM cohort				-- only keep the first diagnoses, no duplicate admissions
		WHERE cohort.rownum = '1';
		
		RETURN QUERY SELECT * FROM patient_cohort_filtered;
		
	ELSE 
		create or replace view patient_cohort_filtered
		AS		
		-- 1. step filter for relevant admissions
		with cohort as
		( 
		SELECT
			patient_cohort_with_icd.hadm_id,			
			patient_cohort_with_icd.icustay_id,		
			patient_cohort_with_icd.intime,			
			patient_cohort_with_icd.outtime,			
			patient_cohort_with_icd.los,
			patient_cohort_with_icd.icustays_count,
			patient_cohort_with_icd.age,
			patient_cohort_with_icd.patientweight,
			patient_cohort_with_icd.dob,
			patient_cohort_with_icd.dod,
			patient_cohort_with_icd.death_in_hosp,
			patient_cohort_with_icd.death_3_days,
			patient_cohort_with_icd.death_30_days,
			patient_cohort_with_icd.death_365_days,				
			patient_cohort_with_icd.gender,			
			patient_cohort_with_icd.ethnicity,		
			patient_cohort_with_icd.first_service,	
			patient_cohort_with_icd.dbsource,		
			patient_cohort_with_icd.subject_id,	
			patient_cohort_with_icd.seq_num,			
			patient_cohort_with_icd.icd9_code,		
			patient_cohort_with_icd.all_icd9_codes,
			ROW_NUMBER() OVER (PARTITION BY 
								patient_cohort_with_icd.hadm_id, 
								patient_cohort_with_icd.icustay_id, 
								patient_cohort_with_icd.subject_id
								ORDER BY 
								patient_cohort_with_icd.hadm_id, 
								patient_cohort_with_icd.icustay_id, 
								patient_cohort_with_icd.subject_id,
							   	patient_cohort_with_icd.seq_num
			) rownum
		FROM patient_cohort_with_icd
		ORDER BY patient_cohort_with_icd.hadm_id, patient_cohort_with_icd.seq_num
		)
		SELECT 
			cohort.hadm_id,			
			cohort.icustay_id,		
			cohort.intime,			
			cohort.outtime,			
			cohort.los,
			cohort.icustays_count,
			cohort.age,
			cohort.patientweight,
			cohort.dob,
			cohort.dod,
			cohort.death_in_hosp,
			cohort.death_3_days,
			cohort.death_30_days,
			cohort.death_365_days,				
			cohort.gender,			
			cohort.ethnicity,		
			cohort.first_service,	
			cohort.dbsource,		
			cohort.subject_id,		
			cohort.seq_num,
			cohort.icd9_code,		
			cohort.all_icd9_codes
		FROM cohort				-- only keep the first diagnoses, no duplicate admissions
		WHERE cohort.icd9_code IN (SELECT icd9_code FROM temp_icd9_codes_selected) 
		AND cohort.rownum = min(cohort.rownum);
		
		RETURN QUERY SELECT * FROM patient_cohort_filtered;

	END IF;

end; $body$

