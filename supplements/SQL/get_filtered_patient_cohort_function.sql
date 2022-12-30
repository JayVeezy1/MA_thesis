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
		all_icd9_codes 				varchar[]
) 

language plpgsql
as $body$
declare
v_selected_code varchar(10);
	
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
	
	RETURN QUERY SELECT * FROM temp_filtered_patient_cohort;

end; $body$
