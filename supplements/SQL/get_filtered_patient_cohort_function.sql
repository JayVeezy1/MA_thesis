create or replace function public.get_filtered_patient_cohort(icd9_selected_list varchar[])		
returns table (
		icustay_id					integer,
		hadm_id						integer,
		subject_id					integer,
		intime						timestamp without time zone,
		outtime						timestamp without time zone,
		los_hours					numeric,
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
		stroke_type					text,
		all_icd9_codes 				varchar[],
		hypertension_flag			int,				-- flag columns must be listed here at the end, but can be moved further to the front in create_transposed_patient
		diabetes_flag				int,
		cancer_flag					int,
		obesity_flag				int,
		drug_abuse_flag				int
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
v_icd9_code				varchar(10);
	
begin
	-- 1. step create list from input string
	if (SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename  = 'temp_icd9_codes_selected')) then	
		DROP TABLE temp_icd9_codes_selected CASCADE;		-- CASCADE needed because view patient_cohort_filtered is related
	end if;
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
				-- TODO: add stroke-type column here with 
				-- hemorrhagic: 430, 431, 432, 4329,
				-- ischemic (+TIA): 433, 4330, 4331, 4332, 434, 4340, 43400, 43401, 4341, 43411, 435, 4350, 4351, 4353, 4359, 436
				-- other (+late_effects_of_stroke): 437, 4370, 4371, 4372, 4373, 4374, 438, 4381, 43811, 4382, 43820, 4383, 4384, 4385, 4388, 43882, 43885
				case when icd_cohort.icd9_code = '430' or icd_cohort.icd9_code = '431' or icd_cohort.icd9_code = '432' or icd_cohort.icd9_code = '4329' 
						then 'hemorrhagic'
					 when icd_cohort.icd9_code = '433' or icd_cohort.icd9_code = '4330' or icd_cohort.icd9_code = '4331' or icd_cohort.icd9_code = '4332' 
						or icd_cohort.icd9_code = '434' or icd_cohort.icd9_code = '4340' or icd_cohort.icd9_code = '43400' 
						or icd_cohort.icd9_code = '43401' or icd_cohort.icd9_code = '4341' or icd_cohort.icd9_code = '43411' 
						or icd_cohort.icd9_code = '435' or icd_cohort.icd9_code = '4350' or icd_cohort.icd9_code = '4351'
						or icd_cohort.icd9_code = '4353' or icd_cohort.icd9_code = '4359' or icd_cohort.icd9_code = '436'
					 	then 'ischemic'
			  		 else 'other_stroke' end 
					 as stroke_type, 
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
	if (SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename  = 'temp_filtered_patient_cohort')) then	
		DROP TABLE public.temp_filtered_patient_cohort;		
	end if;
	
	IF (SELECT COUNT(*) FROM temp_icd9_codes_selected) = 0 THEN	
		raise notice 'Creating temp_filtered_patient_cohort with no selected icd9_codes.';
		CREATE TABLE public.temp_filtered_patient_cohort AS
		SELECT * FROM public.patient_cohort_with_icd
		WHERE patient_cohort_with_icd.seq_num = '1';-- only keep first diagnosis -> no duplicate icustays
	ELSE
		raise notice 'Creating temp_filtered_patient_cohort with selected icd9_codes.';
		CREATE TABLE public.temp_filtered_patient_cohort AS
		SELECT * FROM public.patient_cohort_with_icd
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
		ADD COLUMN drug_abuse_flag				integer;
	UPDATE public.temp_filtered_patient_cohort 
	SET 
		hypertension_flag = 0,
		diabetes_flag = 0,
		cancer_flag = 0,
		obesity_flag = 0,
		drug_abuse_flag = 0;
	
 	FOR v_patient_record in (select temp_filtered_patient_cohort.icustay_id from public.temp_filtered_patient_cohort) LOOP
		-- reset _flags to 0
		v_hypertension_flag := 0;
		v_diabetes_flag := 0;
		v_cancer_flag := 0;
		v_obesity_flag := 0;
		v_drug_abuse_flag := 0;
			-- get the flags for each patient
			FOREACH v_icd9_code in array (select temp_filtered_patient_cohort.all_icd9_codes 
								from public.temp_filtered_patient_cohort 
								where temp_filtered_patient_cohort.icustay_id = v_patient_record.icustay_id) LOOP
				if (SELECT v_icd9_code = ANY ('{4011, 4019, 40210, 40290, 40410, 40490, 4051, 4059}'::varchar[])) then			-- hypertension_icd9_list = '{430,412,413}'
					v_hypertension_flag := 1;
				elsif (SELECT v_icd9_code = ANY ('{25000, 25010, 25020, 25030, 25040, 25050, 25060, 25070, 25090}'::varchar[])) then			-- diabetes_icd9_list = '{510,512,513}'
					v_diabetes_flag := 1;
				elsif (SELECT v_icd9_code = ANY ('{abc, 123}'::varchar[])) then			-- TEST: cancer_icd9_list = '{abc}'
					v_cancer_flag := 1;
				elsif (SELECT v_icd9_code = ANY ('{abc, 123}'::varchar[])) then			-- TEST -- TODO: add correct flag icd9 codes
					v_obesity_flag := 1;
				elsif (SELECT v_icd9_code = ANY ('{abc, 123}'::varchar[])) then			-- TEST -- TODO: add correct flag icd9 codes
					v_drug_abuse_flag := 1;
				end if;	
			END LOOP;	

		-- change the flag values for the patient								
		UPDATE public.temp_filtered_patient_cohort 
		SET 
			hypertension_flag = v_hypertension_flag,
			diabetes_flag = v_diabetes_flag,
			cancer_flag = v_cancer_flag,
			obesity_flag = v_obesity_flag,
			drug_abuse_flag = v_drug_abuse_flag
		WHERE temp_filtered_patient_cohort.icustay_id = v_patient_record.icustay_id;
	END LOOP;
	
	RETURN QUERY SELECT * FROM temp_filtered_patient_cohort;

end; $body$
