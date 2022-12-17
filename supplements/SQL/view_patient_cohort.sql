-- reference: https://github.com/alistairewj/sepsis3-mimic/blob/master/query/tbls/cohort.sql
-- SET search_path TO mimiciii;  		-- This command is only needed once for the creation of the view in the mimiciii schema

create or replace view patient_cohort_view
AS
	with serv as
	(
		select services.hadm_id, curr_service
		, ROW_NUMBER() over (partition by services.hadm_id order by transfertime) as rn
		from services
	)
	, t1 as
	(
	select ie.icustay_id, ie.hadm_id
		, ie.intime, ie.outtime
		, round((cast(adm.admittime as date) - cast(pat.dob as date)) / 365.242, 4) as age
		, pat.gender
		, adm.ethnicity
		, ie.dbsource
		-- used to get first ICUSTAY_ID
		, ROW_NUMBER() over (partition by ie.subject_id order by ie.intime) as rn
		-- exclusions
		, s.curr_service as first_service
		, adm.HAS_CHARTEVENTS_DATA
		-- suspicion of infection using POE / spoe.suspected_infection_time was removed, not needed as it was derived only for special case of sepsis-prediction
	from icustays ie
	inner join admissions adm
		on ie.hadm_id = adm.hadm_id
	inner join patients pat				
		on ie.subject_id = pat.subject_id
	left join serv s
		on ie.hadm_id = s.hadm_id
		and s.rn = 1
	)
	-- FINAL SELECT FOR TABLE --
	select
		t1.hadm_id, t1.icustay_id
	  , t1.intime, t1.outtime
	  -- set de-identified ages to median of 91.4
	  , case when t1.age > 89 then 91.4 else t1.age end as age
	  , t1.gender
	  , t1.ethnicity
	  , t1.first_service
	  , t1.dbsource
	  -- exclusions
	  , case when t1.rn = 1 then 0 else 1 end as exclusion_secondarystay
	  , case when t1.age <= 16 then 1 else 0 end as exclusion_nonadult
	  , case when t1.first_service in ('CSURG','VSURG','TSURG') then 1 else 0 end as exclusion_csurg
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
			  or t1.first_service in ('CSURG','VSURG','TSURG')
			  or t1.HAS_CHARTEVENTS_DATA = 0
			  or t1.intime is null
			  or t1.outtime is null
			  or t1.dbsource != 'metavision'
			then 1
			else 0 end as excluded
	from t1
	order by t1.icustay_id;
			
end;

-- select * from patient_cohort_view;
