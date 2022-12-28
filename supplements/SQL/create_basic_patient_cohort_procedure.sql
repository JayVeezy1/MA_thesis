create or replace procedure public.create_basic_patient_cohort_view()		
/*
returns view (
		hadm_id						integer,
		icustay_id					integer,
		intime						timestamp without time zone,
		outtime						timestamp without time zone,
		los 						double precision,
		icustays_count				bigint,
		age 						numeric,
		patientweight				double precision,
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
		exclusion_secondarystay		int,			
		exclusion_nonadult			int,
		exclusion_carevue			int,
		exclusion_bad_data			int,
		excluded					int					
) 
*/
language plpgsql
as $body$
declare
-- variable declaration with ;	
	
begin
	create or replace view public.patient_cohort_view
	AS
	-- reference: https://github.com/alistairewj/sepsis3-mimic/blob/master/query/tbls/cohort.sql
		with serv as
		(
			select services.hadm_id, curr_service
			, ROW_NUMBER() over (partition by services.hadm_id order by transfertime) as rn
			from mimiciii.services
		),
		input_weight as
		(
			SELECT 		
				inputevents_mv.subject_id,
				inputevents_mv.patientweight
			FROM mimiciii.inputevents_mv
		),	
		counter as(					-- count all icustays for one subject
				SELECT 		
					patients.subject_id,
					count(patients.subject_id) as icustays_count
				FROM mimiciii.patients
				left join mimiciii.icustays
					on icustays.subject_id = patients.subject_id
				GROUP BY patients.subject_id
				ORDER BY patients.subject_id
		) 
		, t1 as
		(
		select ie.icustay_id, ie.hadm_id
			, ie.intime, ie.outtime, ie.los
			, round((cast(ie.intime as date) - cast(pat.dob as date)) / 365.242, 4) as age
			, input_weight.patientweight
			-- adding death related features
			, cast(pat.dob as date) as dob														
			, cast(pat.dod as date) as dod																					-- date works as general death-flag (>365 days)
			, case when (cast(ie.outtime as date) - cast(pat.dod as date)) >= 0 then 1 else 0 end as death_in_hosp			-- flag for in_hosp date exists
			, case when (cast(pat.dod as date) - cast(ie.intime as date)) <= 3 then 1 else 0 end as death_3_days			-- flag for 3 days after admission
			, case when (cast(pat.dod as date) - cast(ie.intime as date)) <= 30 then 1 else 0 end as death_30_days			-- flag for 30 days after admission
			, case when (cast(pat.dod as date) - cast(ie.intime as date)) <= 365 then 1 else 0 end as death_365_days		-- flag for 365 days after admission
			, pat.gender
			, adm.ethnicity
			, ie.dbsource
			-- used to get first ICUSTAY_ID
			, ROW_NUMBER() over (partition by ie.subject_id order by ie.intime) as rn
			-- exclusions
			, s.curr_service as first_service
			, adm.HAS_CHARTEVENTS_DATA
			, counter.icustays_count

		from mimiciii.icustays ie
		inner join mimiciii.admissions adm
			on ie.hadm_id = adm.hadm_id
		inner join mimiciii.patients pat				
			on ie.subject_id = pat.subject_id
		left join serv s
			on ie.hadm_id = s.hadm_id
			and s.rn = 1
		left join counter
			on counter.subject_id = pat.subject_id
		left join input_weight
			on input_weight.subject_id = pat.subject_id
		)
		-- FINAL SELECT FOR TABLE --
		select
		    t1.hadm_id
		  , t1.icustay_id
		  , t1.intime
		  , t1.outtime
		  , t1.los
		  , t1.icustays_count
		  -- set de-identified ages to median of 91.4
		  , case when t1.age > 89 then 91.4 else t1.age end as age
		  , t1.patientweight
		  , t1.dob
		  , t1.dod
		  , t1.death_in_hosp
		  , t1.death_3_days
		  , t1.death_30_days
		  , t1.death_365_days
		  , t1.gender
		  , t1.ethnicity
		  , t1.first_service
		  , t1.dbsource
		  -- exclusions
		  , case when t1.rn = 1 then 0 else 1 end as exclusion_secondarystay
		  , case when t1.age <= 16 then 1 else 0 end as exclusion_nonadult
		  -- , case when t1.first_service in ('CSURG','VSURG','TSURG') then 1 else 0 end as exclusion_csurg
		  , case when t1.dbsource != 'metavision' then 1 else 0 end as exclusion_carevue
		  , case when t1.HAS_CHARTEVENTS_DATA = 0 then 1
				 when t1.intime is null then 1
				 when t1.outtime is null then 1
			  else 0 end as exclusion_bad_data
		  -- the above flags are used to summarize patients excluded
		  -- below flag is used to actually exclude patients in future queries
		  , case when
					 t1.rn != 1
				  or t1.age <= 16
				  -- or t1.first_service in ('CSURG','VSURG','TSURG')		-- surgery patients should not be considered
				  or t1.HAS_CHARTEVENTS_DATA = 0
				  or t1.intime is null
				  or t1.outtime is null
				  or t1.dbsource != 'metavision'
				then 1
				else 0 end as excluded
		from t1
		order by t1.icustay_id;
			
end; $body$

