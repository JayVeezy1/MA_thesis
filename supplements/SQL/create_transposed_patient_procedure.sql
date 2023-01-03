-- https://stackoverflow.com/questions/12879672/dynamically-generate-columns-for-crosstab-in-postgresql

create or replace procedure public.create_transposed_patient(input_icustay_id integer, selected_itemids_list varchar[])		
language plpgsql
as $body$
declare
	rec 				RECORD;
	v_patient_record 	RECORD;
	str 				text;
	v_day 				int;
	v_first_intime 		timestamp without time zone;
	v_oasis_record		RECORD;
	
begin
	raise notice 'Creating transposed_patient for icustay_id: %', input_icustay_id;

-- Step 1: Get all_events_view temporarily into temp_single_patient table
	DROP TABLE IF EXISTS public.temp_single_patient;				
    CREATE TABLE public.temp_single_patient AS
    SELECT * FROM public.get_all_events_view(input_icustay_id, selected_itemids_list);		-- selected_itemids_list can be '{}', then all available itemids will be selected

	str := ' "charttime" timestamp without time zone, ';			-- text = field type , not icustay_id in here, because it would group all rows
   	FOR rec IN SELECT DISTINCT label								-- looping to get column heading string
        FROM temp_single_patient									
		ORDER BY temp_single_patient.label
	LOOP
    	str :=  str || '"' || rec.label || '" text' ||',';			-- str is used for defintion of table below
    END LOOP;
    str:= substring(str, 0, length(str));							-- remove last comma
	raise notice 'Selected labels for header str: %', str;

-- Step 2: Dynamically create a temp_table with transposed row->column for selected labels
    EXECUTE 												 
	'CREATE EXTENSION IF NOT EXISTS tablefunc;
    DROP TABLE IF EXISTS temp_transposed_patient;
    CREATE TABLE temp_transposed_patient AS
    SELECT *
    FROM crosstab(''select temp_single_patient.charttime, temp_single_patient.label, temp_single_patient.value 
					from temp_single_patient order by 1'',
                 ''SELECT DISTINCT temp_single_patient.label FROM temp_single_patient ORDER BY 1'')
    AS final_result ('|| str ||')';
	
-- Setp 3: Join the chart_events data with the general patient data
	DELETE FROM temp_transposed_patient WHERE temp_transposed_patient.charttime = '0001-01-01 00:00:01';	-- remove DUMMY values, those were added in get_all_events_view
	
	-- check if not exists (should never be the case, always call single transposed_patient after having created temp_filtered_patient_cohort)
	if not (SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename  = 'temp_filtered_patient_cohort')) then	
		raise notice 'WARNING: temp_filtered_patient_cohort should have already existed for creation of transposed_patient.';
		CREATE TABLE public.temp_filtered_patient_cohort AS
		SELECT * FROM get_filtered_patient_cohort('{}');		
	end if;
	
	SELECT * FROM temp_filtered_patient_cohort 
	WHERE temp_filtered_patient_cohort.icustay_id = input_icustay_id 
	INTO v_patient_record;
	
	ALTER TABLE public.temp_transposed_patient
		ADD COLUMN icustay_id					int,
		ADD COLUMN hadm_id						int, 
		ADD COLUMN subject_id					int, 
		ADD COLUMN intime						timestamp without time zone, 
		ADD COLUMN outtime						timestamp without time zone, 
		ADD COLUMN los_hours 					numeric,
		ADD COLUMN day_on_icu 					int,				-- day_on_icu
		ADD COLUMN icustays_count				bigint,
		ADD COLUMN age							numeric,
		ADD COLUMN patientweight				double precision,
		ADD COLUMN gender						varchar(5),
		ADD COLUMN ethnicity					varchar(200),
		ADD COLUMN admission_type				varchar(50),
		ADD COLUMN discharge_location 			varchar(50),
		ADD COLUMN insurance 					varchar(255),
		ADD COLUMN language 					varchar(10),
		ADD COLUMN religion						varchar(50),
		ADD COLUMN marital_status				varchar(50),
		ADD COLUMN diagnosis_text				text,
		ADD COLUMN dob 							date,
		ADD COLUMN dod 							date,
		ADD COLUMN death_in_hosp 				int,
		ADD COLUMN death_3_days 				int,
		ADD COLUMN death_30_days 				int,
		ADD COLUMN death_180_days 				int,
		ADD COLUMN death_365_days 				int,
		-- oasis score
		ADD COLUMN OASIS						int,
		ADD COLUMN OASIS_PROB					numeric,		-- oasis derived in-hospitality death-probability
	    ADD COLUMN preiculos					interval, 
	    ADD COLUMN gcs							double precision, 
	    ADD COLUMN mechvent						int, 			-- mechanical ventilation
	    ADD COLUMN electivesurgery 				int,
		-- comorbidities
		ADD COLUMN stroke_type					text,
		ADD COLUMN hypertension_flag			int,
		ADD COLUMN diabetes_flag				int,
		ADD COLUMN cancer_flag					int,
		ADD COLUMN obesity_flag					int,
		ADD COLUMN drug_abuse_flag				int,
		ADD COLUMN sepsis_flag					int,
		ADD COLUMN icd9_code					varchar(10),
		ADD COLUMN all_icd9_codes 				varchar[];	
		
	-- Inserting general patient values
	UPDATE public.temp_transposed_patient 
	SET 
		icustay_id = input_icustay_id,
		hadm_id = v_patient_record.hadm_id,
		subject_id = v_patient_record.subject_id,
		intime = v_patient_record.intime,
		outtime = v_patient_record.outtime,
		los_hours = v_patient_record.los_hours,
		icustays_count = v_patient_record.icustays_count,
		age = v_patient_record.age,
		patientweight = v_patient_record.patientweight,
		gender = v_patient_record.gender,
		ethnicity = v_patient_record.ethnicity,
		admission_type = v_patient_record.admission_type,
		discharge_location = v_patient_record.discharge_location,
		insurance = v_patient_record.insurance,
		language = v_patient_record.language,
		religion = v_patient_record.religion,
		marital_status = v_patient_record.marital_status,
		diagnosis_text = v_patient_record.diagnosis_text,
		dob = v_patient_record.dob,	
		dod = v_patient_record.dod,						
		death_in_hosp = v_patient_record.death_in_hosp, 				
		death_3_days = v_patient_record.death_3_days, 				
		death_30_days = v_patient_record.death_30_days,  
		death_180_days = v_patient_record.death_180_days, 
		death_365_days = v_patient_record.death_365_days, 
		hypertension_flag = v_patient_record.hypertension_flag,
		diabetes_flag = v_patient_record.diabetes_flag,
		cancer_flag = v_patient_record.cancer_flag,
		obesity_flag = v_patient_record.obesity_flag,
		drug_abuse_flag = v_patient_record.drug_abuse_flag,
		sepsis_flag = v_patient_record.sepsis_flag,
		stroke_type = v_patient_record.stroke_type,
		icd9_code = v_patient_record.icd9_code,
		all_icd9_codes = v_patient_record.all_icd9_codes;

	-- inserting day_on_icu
	SELECT min(intime) FROM mimiciii.icustays WHERE mimiciii.icustays.icustay_id = v_patient_record.icustay_id INTO v_first_intime;	
	UPDATE public.temp_transposed_patient 
	SET 
		day_on_icu = cast(Round(EXTRACT(epoch FROM temp_transposed_patient.charttime - v_first_intime)/3600/24, 0) + 1 as integer)	
	WHERE temp_transposed_patient.icustay_id = v_patient_record.icustay_id;

	-- inserting OASIS values for correct day_on_icu
	FOR v_day in 1..(SELECT max(day_on_icu) 
					 FROM public.temp_transposed_patient
					 WHERE temp_transposed_patient.icustay_id = v_patient_record.icustay_id) LOOP
 
		-- use a view to add day_on_icu also here
		with oasis_1 as (
			SELECT 
				*,
				ROW_NUMBER() OVER (PARTITION BY oasis.icustay_id ORDER BY oasis.icustay_id) as day_on_icu
			FROM mimiciii.oasis
			WHERE oasis.icustay_id = v_patient_record.icustay_id
		)
		SELECT 
			oasis_1.OASIS,
			oasis_1.OASIS_PROB,
			oasis_1.preiculos,
			oasis_1.gcs,
			oasis_1.mechvent,
			oasis_1.electivesurgery 
		FROM oasis_1
		WHERE oasis_1.icustay_id = v_patient_record.icustay_id 
		AND oasis_1.day_on_icu = v_day								-- filter for the selected v_day
		INTO v_oasis_record;
	
		UPDATE public.temp_transposed_patient 
		SET  
			OASIS = v_oasis_record.OASIS,		
			OASIS_PROB = v_oasis_record.OASIS_PROB,
			preiculos = v_oasis_record.preiculos,
			gcs = v_oasis_record.gcs,
			mechvent = v_oasis_record.mechvent,
			electivesurgery = v_oasis_record.electivesurgery		
		WHERE day_on_icu = v_day;
		END LOOP;

end; $body$
