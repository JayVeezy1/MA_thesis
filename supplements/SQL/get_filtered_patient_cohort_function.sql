create or replace function public.get_filtered_patient_cohort(icd9_selected_list varchar[])		
returns table (
		icustay_id					integer,
		hadm_id						integer,
		subject_id					integer,
		intime						timestamp without time zone,
		outtime						timestamp without time zone,
		los_hours					numeric,
		dbsource					varchar(20),
		icustays_count				bigint,
		age 						numeric,
		patientweight				double precision,
		dob 						date,
		dod 						date,
		death_in_hosp 				int,
		death_3_days 				int,
		death_30_days 				int,
		death_180_days 				int,
		death_365_days 				int,				
		gender						varchar(5),
		ethnicity					varchar(200),
		admission_type				varchar(50),
		discharge_location 			varchar(50),
		insurance 					varchar(255),
		language					varchar(10),
		religion					varchar(50),
		marital_status				varchar(50),
		diagnosis_text				text,
		seq_num						int,
		icd9_code					varchar(10),
		stroke_type					varchar(15),
		infarct_type				varchar(15),
		all_icd9_codes 				varchar[],
		hypertension_flag			int,				-- flag columns must be listed here at the end, but can be moved further to the front in create_transposed_patient
		diabetes_flag				int,
		cancer_flag					int,
		obesity_flag				int,
		drug_abuse_flag				int,
		sepsis_flag					int
	/*
		, total_days_on_icu			int,
		OASIS						int,
		OASIS_PROB					numeric,		-- oasis derive in-hospitality death-probability
	    preiculos					interval, 
	    gcs							double precision, 
	    mechvent					int, 			-- mechanical ventilation
	    electivesurgery 			int				-- if not elective, then emergency surgery -> bad score
	*/
) 

language plpgsql
as $body$
declare
v_selected_code 		varchar(10);
v_patient_record 		RECORD;
v_hypertension_flag		int := 0;
v_diabetes_flag		  	int := 0;
v_cancer_flag			int := 0;
v_obesity_flag 			int := 0;
v_drug_abuse_flag 		int := 0;
v_sepsis_flag			int := 0;
v_icd9_code				varchar(10);
v_icustay_id            int;
v_days_counter			int;
v_oasis_record			RECORD;

begin
	-- 1. step create list from input string
	DROP TABLE IF EXISTS temp_icd9_codes_selected CASCADE;		-- CASCADE needed because view patient_cohort_filtered is related
	CREATE TABLE temp_icd9_codes_selected(icd9_codes varchar(10));
	
	FOREACH v_selected_code IN ARRAY icd9_selected_list LOOP
		INSERT INTO temp_icd9_codes_selected (icd9_codes)
		VALUES (v_selected_code);
	END LOOP;
	
	-- 2. step create view patient_cohort_with_icd	
	CALL create_basic_patient_cohort_view();
	
	create or replace view patient_cohort_with_icd
	AS
		with diagnoses as
		(
			select * from get_all_diagnoses()
		), icd_cohort as
		(
			select 				
				basic_cohort.icustay_id,
				basic_cohort.hadm_id,
				basic_cohort.subject_id,
				basic_cohort.intime,
				basic_cohort.outtime,
				basic_cohort.los_hours,
				basic_cohort.dbsource,
				basic_cohort.icustays_count,
				basic_cohort.age,
				basic_cohort.patientweight,
				basic_cohort.dob,
				basic_cohort.dod,
				basic_cohort.death_in_hosp,
				basic_cohort.death_3_days,
				basic_cohort.death_30_days,
				basic_cohort.death_180_days,
				basic_cohort.death_365_days,
				basic_cohort.gender,
				basic_cohort.ethnicity,
				basic_cohort.admission_type,
				basic_cohort.discharge_location,
				basic_cohort.insurance,
				basic_cohort.language,
				basic_cohort.religion,
				basic_cohort.marital_status,
				basic_cohort.diagnosis_text,
				basic_cohort.excluded,
				diag.seq_num,
				diag.icd9_code,
				diag.stroke_type,
				diag.infarct_type,
				diag.all_icd9_codes	
			from patient_cohort_view basic_cohort
			inner join diagnoses diag
				on basic_cohort.hadm_id = diag.hadm_id		
		)
		select
				icd_cohort.icustay_id,
				icd_cohort.hadm_id,
				icd_cohort.subject_id,
				icd_cohort.intime,
				icd_cohort.outtime,
				icd_cohort.los_hours,
				icd_cohort.dbsource,
				icd_cohort.icustays_count,
				icd_cohort.age,
				icd_cohort.patientweight,
				icd_cohort.dob,
				icd_cohort.dod,
				icd_cohort.death_in_hosp,
				icd_cohort.death_3_days,
				icd_cohort.death_30_days,
				icd_cohort.death_180_days,
				icd_cohort.death_365_days,
				icd_cohort.gender,
				icd_cohort.ethnicity,
				icd_cohort.admission_type,
				icd_cohort.discharge_location,
				icd_cohort.insurance,
				icd_cohort.language,
				icd_cohort.religion,
				icd_cohort.marital_status,
				icd_cohort.diagnosis_text,
				icd_cohort.seq_num,
				icd_cohort.icd9_code,
				icd_cohort.stroke_type,
				icd_cohort.infarct_type,
				icd_cohort.all_icd9_codes
		from icd_cohort 
		where icd_cohort.excluded = 0					-- remove all 'excluded' entries
		and los_hours > 24								-- remove all icustays with < 24 hours
		order by icd_cohort.icustay_id, icd_cohort.seq_num;		
	-- select count(*) from  patient_cohort_with_icd;
	-- join leads to 705921 entries, but in diagnoses only 600.000 hadm_id entries 
	-- because in cohort_view some hadm_ids are duplicate, because it was 1 hadm but multiple times icustay_id
	-- solution: with "excluded = 0" count = 513035 and independent on join-type (right, left, inner) -> always same matches.
	

	-- 3. step depending on filter select icd9_codes and single diagnosis 
	-- this is actually 'creator' instead of 'getter' function. But it works.
	DROP TABLE IF EXISTS public.temp_filtered_patient_cohort;		
	
	IF (SELECT COUNT(*) FROM temp_icd9_codes_selected) = 0 THEN	
		raise notice 'Creating temp_filtered_patient_cohort with no selected icd9_codes.';
		CREATE TABLE public.temp_filtered_patient_cohort AS
		SELECT * FROM public.patient_cohort_with_icd
		WHERE patient_cohort_with_icd.seq_num = '1';-- only keep first diagnosis -> no duplicate icustays
	ELSE
		raise notice 'Creating temp_filtered_patient_cohort with selected icd9_codes.';
		
		-- FILTERING of patients happens here
		CREATE TABLE public.temp_filtered_patient_cohort AS
		SELECT * FROM public.patient_cohort_with_icd			-- patient_cohort_with_icd has a row per diagnosis
		 WHERE patient_cohort_with_icd.icd9_code IN (SELECT icd9_codes FROM public.temp_icd9_codes_selected)			-- filter for icd9_codes here
		 AND concat(cast(patient_cohort_with_icd.icustay_id as varchar), cast(patient_cohort_with_icd.seq_num as varchar))
			IN (SELECT concat(cast(patient_cohort_with_icd.icustay_id as varchar), cast(min(patient_cohort_with_icd.seq_num) as varchar))
				FROM public.patient_cohort_with_icd 
				WHERE public.patient_cohort_with_icd.icd9_code IN (SELECT icd9_codes FROM public.temp_icd9_codes_selected)
				GROUP BY patient_cohort_with_icd.icustay_id);		-- only keep the first fitting diagnoses, no duplicate icustays
	END IF;
	
	-- 4. step add comorbidities to the temp_filtered_patient_cohort
		-- add the flags as new columns into the final cohort table
	ALTER TABLE public.temp_filtered_patient_cohort
		ADD COLUMN hypertension_flag			integer,
		ADD COLUMN diabetes_flag				integer,
		ADD COLUMN cancer_flag					integer,
		ADD COLUMN obesity_flag					integer,
		ADD COLUMN drug_abuse_flag				integer,
		ADD COLUMN sepsis_flag					integer;

	UPDATE public.temp_filtered_patient_cohort 
	SET 
		hypertension_flag = 0,
		diabetes_flag = 0,
		cancer_flag = 0,
		obesity_flag = 0,
		drug_abuse_flag = 0;
		sepsis_flag = 0;
	
 	FOR v_patient_record in (select temp_filtered_patient_cohort.icustay_id from public.temp_filtered_patient_cohort) LOOP
		-- reset _flags to 0
		v_hypertension_flag := 0;
		v_diabetes_flag := 0;
		v_cancer_flag := 0;
		v_obesity_flag := 0;
		v_drug_abuse_flag := 0;
		v_sepsis_flag := 0;
		
			-- get the flags for each patient
			FOREACH v_icd9_code in array (select temp_filtered_patient_cohort.all_icd9_codes 
								from public.temp_filtered_patient_cohort 
								where temp_filtered_patient_cohort.icustay_id = v_patient_record.icustay_id) LOOP
				if (SELECT v_icd9_code = ANY (SELECT codes FROM public.comorbidity_codes WHERE category = 'hypertension')) then			-- hypertension_icd9_list = '{430,412,413}'
					v_hypertension_flag := 1;
				elsif (SELECT v_icd9_code = ANY (SELECT codes FROM public.comorbidity_codes WHERE category = 'diabetes')) then			-- diabetes_icd9_list = '{510,512,513}'
					v_diabetes_flag := 1;
				elsif (SELECT v_icd9_code = ANY (SELECT codes FROM public.comorbidity_codes WHERE category = 'cancer')) then			-- cancer_icd9_list = '{}'
					v_cancer_flag := 1;
				elsif (SELECT v_icd9_code = ANY (SELECT codes FROM public.comorbidity_codes WHERE category = 'obesity')) then			-- obesity_icd9_list = '{2780, 27800, 27801, 27802, 2781}'
					v_obesity_flag := 1;
				elsif (SELECT v_icd9_code = ANY (SELECT codes FROM public.comorbidity_codes WHERE category = 'drug_abuse')) then		-- drug_abuse_icd9_list = '{}'
					v_drug_abuse_flag := 1;
				elsif (SELECT v_icd9_code = ANY (SELECT codes FROM public.comorbidity_codes WHERE category = 'sepsis')) then		-- drug_abuse_icd9_list = '{}'
					v_sepsis_flag := 1;
				end if;	
			END LOOP;	

		-- change the flag values for the patient								
		UPDATE public.temp_filtered_patient_cohort 
		SET 
			hypertension_flag = v_hypertension_flag,
			diabetes_flag = v_diabetes_flag,
			cancer_flag = v_cancer_flag,
			obesity_flag = v_obesity_flag,
			drug_abuse_flag = v_drug_abuse_flag,
			sepsis_flag = v_sepsis_flag
		WHERE temp_filtered_patient_cohort.icustay_id = v_patient_record.icustay_id;
	END LOOP;
	
	/* Removed because: this step raises the time needed for this function from 30 seconds to 4 minutes, also total_days_on_icu not really needed at all
	-- 5. step add the columns related to the OASIS Score (the views that were used for this are from a github repository in the references)	
	ALTER TABLE public.temp_filtered_patient_cohort
		-- adding oasis related columns here
		-- but leave out vital signs from the oasis-view because their calculation will be done in Python
		ADD COLUMN total_days_on_icu			int,
		ADD COLUMN OASIS						int,
		ADD COLUMN OASIS_PROB					numeric,		-- oasis derived in-hospitality death-probability
	    ADD COLUMN preiculos					interval, 
	    ADD COLUMN gcs							double precision, 
	    ADD COLUMN mechvent						int, 			-- mechanical ventilation
	    ADD COLUMN electivesurgery 				int;
	
	FOR v_icustay_id IN (SELECT temp_filtered_patient_cohort.icustay_id FROM public.temp_filtered_patient_cohort) LOOP
		
		SELECT 
			count(oasis.icustay_id) 				-- TODO: might not be needed here at all
		FROM mimiciii.oasis WHERE mimiciii.oasis.icustay_id = v_icustay_id 
		INTO v_days_counter;
		
		SELECT 
			mimiciii.oasis.OASIS,
			mimiciii.oasis.OASIS_PROB,
			mimiciii.oasis.preiculos,
			mimiciii.oasis.gcs,
			mimiciii.oasis.mechvent,
			mimiciii.oasis.electivesurgery 
		FROM mimiciii.oasis WHERE mimiciii.oasis.icustay_id = v_icustay_id 
		INTO v_oasis_record;
		
		UPDATE public.temp_filtered_patient_cohort 
			SET 
				total_days_on_icu = v_days_counter,
				OASIS = v_oasis_record.OASIS,
				OASIS_PROB = v_oasis_record.OASIS_PROB,
				preiculos = v_oasis_record.preiculos,
				gcs = v_oasis_record.gcs,
				mechvent = v_oasis_record.mechvent,
				electivesurgery = v_oasis_record.electivesurgery
			WHERE public.temp_filtered_patient_cohort.icustay_id = v_icustay_id;
			
	END LOOP;
	*/
	
	RETURN QUERY SELECT * FROM public.temp_filtered_patient_cohort;		 

end; $body$
