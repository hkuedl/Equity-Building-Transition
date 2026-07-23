


# HybridIslandingControl


#


#


#


require 'openstudio'
require 'json'
require 'date'

class HybridIslandingControl < OpenStudio::Measure::EnergyPlusMeasure
  EMS_ACTUATABLE_SCHEDULE_TYPES = [
    'Schedule:Compact',
    'Schedule:Year',
    'Schedule:Constant',
    'Schedule:File'
  ].freeze

  PATCH_PREFIX = 'hybrid_island_'.freeze

  PROGRAM_NAME = 'hybrid_island_supervisor'.freeze
  PCM_NAME = 'hybrid_island_supervisor_pcm'.freeze

  ISLAND_SCHEDULE_NAME = 'utility island schedule'.freeze
  CHARGE_SCHEDULE_NAME = 'battery charge cmd schedule'.freeze
  DISCHARGE_SCHEDULE_NAME = 'battery discharge cmd schedule'.freeze
  ISLAND_HVAC_SCHEDULE_NAME = 'island hvac availability schedule'.freeze
  BACKUP_HTG_SCHEDULE_NAME = 'backup htg availability schedule'.freeze
  CONVERTER_NAME = 'battery storage converter'.freeze

  class MissingHybridDerError < StandardError; end


  # Measure metadata


  def name
    'HybridIslandingControl'
  end

  def description
    'Adds hybrid islanding control for ResStock so the building can switch between grid-connected operation and equivalent off-grid/islanded operation using outage scenarios, battery, PV, and EMS control.'
  end

  def modeler_description
    <<~DESC
      This EnergyPlus measure is intended for ResStock/BuildStock upgrades.

      Main logic:
      - When outage_mode=on, the measure automatically resolves the county code,
        looks up county-specific outage windows from JSON, and creates a utility island schedule.
      - During utility-available periods, the building behaves as grid-connected.
      - During outage periods, the building enters an equivalent islanded state:
        selected noncritical loads are shed, HVAC availability is restricted,
        thermostat / sequential fraction schedules can be overridden,
        and battery/PV dispatch is controlled through EMS + ElectricLoadCenter schedules.

      Important note:
      EnergyPlus does not provide a literal "open main breaker" object to disconnect
      the utility in the same way a physical microgrid switch would. Therefore this
      measure approximates off-grid islanding by suppressing loads and utility dependence
      so that purchased electricity should drop toward zero during outage hours.
    DESC
  end


  def cf(str)
    str.to_s.strip.downcase
  end

  def obj_type_name(obj)
    obj.iddObject.name.to_s
  end

  def obj_name(obj)
    s = obj.getString(0)
    return nil unless s.is_initialized

    val = s.get.to_s.strip
    return nil if val.empty?

    val
  end

  def safe_get_string(obj, idx)
    return nil if idx.nil?

    s = obj.getString(idx)
    return nil unless s.is_initialized

    val = s.get.to_s.strip
    return nil if val.empty?

    val
  end

  def schedule_object?(obj)
    obj_type_name(obj).downcase.start_with?('schedule:')
  end

  def generator_object?(obj)
    cf(obj_type_name(obj)).start_with?('generator:')
  end

  def ems_actuatable_schedule_type?(obj_or_type_name)
    type_name = obj_or_type_name.is_a?(String) ? obj_or_type_name : obj_type_name(obj_or_type_name)
    EMS_ACTUATABLE_SCHEDULE_TYPES.map(&:downcase).include?(type_name.to_s.downcase)
  end

  def unique_named_objects(objects)
    dedup = {}
    objects.each do |obj|
      dedup[[cf(obj_type_name(obj)), cf(obj_name(obj))]] = obj
    end
    dedup.keys.sort.map { |k| dedup[k] }
  end

  def make_ems_name(prefix, component_name, idx)
    base = component_name.to_s.gsub(/[^A-Za-z0-9]+/, '_').gsub(/^_+|_+$/, '')
    base = 'obj' if base.empty?
    base = "x_#{base}" unless base[0] =~ /[A-Za-z]/
    base = base[0, 36]
    "#{PATCH_PREFIX}#{prefix}_#{idx}_#{base}"
  end

  def field_index_by_name_fragment(obj, fragment)
    i = 0
    loop do
      idd_field = obj.iddObject.getField(i)
      break unless idd_field.is_initialized

      fname = idd_field.get.name.to_s
      return i if cf(fname).include?(cf(fragment))

      i += 1
    end
    nil
  end

  def field_index_by_name_fragments(obj, fragments)
    Array(fragments).each do |fragment|
      idx = field_index_by_name_fragment(obj, fragment)
      return idx unless idx.nil?
    end
    nil
  end

  def set_field_value_by_name_fragments(obj, fragments, value, required: true)
    idx = field_index_by_name_fragments(obj, fragments)

    if idx.nil?
      raise "Not found #{Array(fragments).join(' / ')} (#{obj_type_name(obj)} / #{obj_name(obj)})" if required
      return false
    end

    obj.setString(idx, value.to_s)
    true
  end

  def distribution_generator_list_name(dist_obj)
    idx = field_index_by_name_fragments(dist_obj, ['Generator List Name'])
    idx = 1 if idx.nil?
    safe_get_string(dist_obj, idx)
  end

  def distribution_inverter_name(dist_obj)
    idx = field_index_by_name_fragments(dist_obj, ['Inverter Name', 'Inverter Object Name'])
    idx = 7 if idx.nil?
    safe_get_string(dist_obj, idx)
  end

  def distribution_storage_name(dist_obj)
    idx = field_index_by_name_fragments(dist_obj, ['Electrical Storage Object Name'])
    idx = 8 if idx.nil?
    safe_get_string(dist_obj, idx)
  end

  def build_storage_converter_idf(max_power_w)
    <<~IDF
      ElectricLoadCenter:Storage:Converter,
        #{CONVERTER_NAME},
        Always On Discrete,
        SimpleFixed,
        0.96,
        #{format('%.1f', max_power_w)};
    IDF
  end

  def build_distribution_idf(core, battery_min_soc:, battery_max_soc:, charge_power_w:, discharge_power_w:)
    <<~IDF
      ElectricLoadCenter:Distribution,
        #{core[:distribution_name]},
        #{core[:generator_list_name]},
        Baseload,
        ,
        ,
        ,
        DirectCurrentWithInverterACStorage,
        #{core[:inverter_name]},
        #{core[:battery_name]},
        ,
        TrackChargeDischargeSchedules,
        ,
        #{CONVERTER_NAME},
        #{format('%.3f', battery_max_soc)},
        #{format('%.3f', battery_min_soc)},
        #{format('%.1f', charge_power_w)},
        #{CHARGE_SCHEDULE_NAME},
        #{format('%.1f', discharge_power_w)},
        #{DISCHARGE_SCHEDULE_NAME};
    IDF
  end

  def add_idf_object(workspace, runner, text)
    loaded = OpenStudio::IdfObject.load(text)
    if loaded.empty?
      runner.registerError("Could not parse the IDF object:\n#{text}")
      return false
    end

    workspace.addObject(loaded.get)
    true
  end

  def empty_optional_controls
    {
      noncritical_eq: [],
      fridge_eq: [],
      interior_lights: [],
      exterior_lights: [],
      bath_fan_schedules: [],
      range_fan_schedules: [],
      water_heater_setpoints: [],
      hvac_fraction_schedules: [],
      heating_setpoint_schedules: [],
      cooling_setpoint_schedules: [],
      hvac_availability_schedules: [],
      backup_htg_availability_schedules: []
    }
  end

  def normalize_optional_controls(controls)
    normalized = empty_optional_controls
    controls = {} if controls.nil?

    normalized.keys.each do |k|
      normalized[k] = Array(controls[k])
    end

    normalized
  end

  def add_idf_objects(workspace, runner, texts)
    texts.each do |txt|
      next if txt.nil? || txt.strip.empty?
      return false unless add_idf_object(workspace, runner, txt)
    end
    true
  end

  def find_object_by_type_and_name(workspace, object_type, object_name)
    workspace.getObjectsByType(object_type.to_IddObjectType).each do |obj|
      return obj if cf(obj_name(obj)) == cf(object_name)
    end
    nil
  end

  def find_named_with_type_prefix(workspace, object_name, prefixes)
    workspace.objects.each do |obj|
      next unless cf(obj_name(obj)) == cf(object_name)
      return obj if prefixes.any? { |p| cf(obj_type_name(obj)).start_with?(cf(p)) }
    end
    nil
  end

  def remove_all_of_type(workspace, object_type)
    workspace.getObjectsByType(object_type.to_IddObjectType).each(&:remove)
  end

  def remove_named_object(workspace, object_type, object_name)
    workspace.getObjectsByType(object_type.to_IddObjectType).each do |obj|
      obj.remove if cf(obj_name(obj)) == cf(object_name)
    end
  end

  def remove_any_named_schedule(workspace, schedule_name)
    workspace.objects.each do |obj|
      next unless schedule_object?(obj)
      obj.remove if cf(obj_name(obj)) == cf(schedule_name)
    end
  end

  def remove_ems_objects_with_prefix(workspace, object_type, prefix = PATCH_PREFIX)
    workspace.getObjectsByType(object_type.to_IddObjectType).each do |obj|
      name = obj_name(obj)
      next if name.nil?
      obj.remove if cf(name).start_with?(cf(prefix))
    end
  end

  def remove_output_variable(workspace, key_value, variable_name)
    workspace.getObjectsByType('Output:Variable'.to_IddObjectType).each do |obj|
      key = safe_get_string(obj, 0)
      var = safe_get_string(obj, 1)
      next unless cf(key) == cf(key_value)
      next unless cf(var) == cf(variable_name)

      obj.remove
    end
  end

  def remove_output_meter(workspace, meter_name)
    workspace.getObjectsByType('Output:Meter'.to_IddObjectType).each do |obj|
      meter = safe_get_string(obj, 0)
      obj.remove if cf(meter) == cf(meter_name)
    end
  end


  def county_fips_to_gisjoin(fips5)
    s = fips5.to_s.strip
    return nil unless s.match?(/^\d{5}$/)

    "G#{s[0, 2]}0#{s[2, 3]}0"
  end

  def normalize_county_code(val)
    s = val.to_s.strip
    return nil if s.empty?

    return s.upcase if s.match?(/^G\d{7}$/i)
    return county_fips_to_gisjoin(s) if s.match?(/^\d{5}$/)

    nil
  end

  def candidate_existing_xml_paths
    dirs = [
      Dir.pwd,
      File.expand_path('..', Dir.pwd)#,


    ].uniq

    paths = []
    dirs.each do |d|
      p = File.join(d, 'in.xml')
      paths << p if File.exist?(p)
    end
    paths.uniq
  end

  def detect_county_from_existing_xml(runner)
    candidate_existing_xml_paths.each do |path|
      begin
        txt = File.read(path)

        txt.scan(/<EPWFilePath>\s*([^<]+)\s*<\/EPWFilePath>/im).flatten.each do |epw_path|
          base = File.basename(epw_path.to_s.strip, '.epw')
          code = normalize_county_code(base)
          unless code.nil?
            runner.registerInfo("Detected county:#{code}(source existing.xml EPWFilePath=#{epw_path})")
            return code
          end
        end

        txt.scan(/<SystemIdentifier\s+id=['"]WeatherStation['"]\s*\/>\s*<Name>\s*([^<]+)\s*<\/Name>/im).flatten.each do |name|
          code = normalize_county_code(name)
          unless code.nil?
            runner.registerInfo("Detected county:#{code}(source existing.xml WeatherStation Name=#{name})")
            return code
          end
        end

        txt.scan(/<Name>\s*([^<]+)\s*<\/Name>/im).flatten.each do |name|
          code = normalize_county_code(name)
          unless code.nil?
            runner.registerInfo("Detected county:#{code}(source existing.xml  Name=#{name})")
            return code
          end
        end
      rescue StandardError => e
        runner.registerWarning("Failed to read existing.xml:#{path};#{e}")
      end
    end

    nil
  end

  def detect_county_from_env(runner)
    %w[COUNTY_CODE BUILDSTOCK_COUNTY GISJOIN COUNTY_FIPS].each do |env_name|
      val = ENV[env_name]
      next if val.nil? || val.strip.empty?

      normalized = normalize_county_code(val)
      next if normalized.nil?

      runner.registerInfo("Detected county:#{normalized}(source ENV[#{env_name}])")
      return normalized
    end
    nil
  end

  def candidate_osw_paths
    [
      File.join(Dir.pwd, 'in.osw'),
      File.join(Dir.pwd, 'out.osw'),
      File.join(Dir.pwd, '..', 'in.osw'),
      File.join(Dir.pwd, '..', 'out.osw'),
      File.join(Dir.pwd, '..', '..', 'in.osw'),
      File.join(Dir.pwd, '..', '..', 'out.osw')
    ].uniq.select { |p| File.exist?(p) }
  end

  def scan_hash_for_county_candidates(node, path = [], out = [])
    case node
    when Hash
      node.each { |k, v| scan_hash_for_county_candidates(v, path + [k.to_s], out) }
    when Array
      node.each_with_index { |v, i| scan_hash_for_county_candidates(v, path + ["[#{i}]"], out) }
    else
      key_path = path.join('.')
      key_cf = cf(key_path)
      val = node.to_s.strip
      return out if val.empty?

      hint = %w[county county_code countyfips fips geoid gisjoin nhgis weather epw].any? { |w| key_cf.include?(w) }

      if val.match?(/^G\d{7}$/i)
        score = hint ? 200 : 100
        out << [score, val.upcase, key_path]
      elsif hint && val.match?(/^\d{5}$/)
        gis = county_fips_to_gisjoin(val)
        out << [180, gis, key_path] unless gis.nil?
      end
    end
    out
  end

  def detect_county_from_osw(runner)
    candidate_osw_paths.each do |path|
      begin
        h = JSON.parse(File.read(path))
        candidates = scan_hash_for_county_candidates(h)
        next if candidates.empty?

        best = candidates.sort_by { |x| [-x[0], x[2].to_s.length] }.first
        runner.registerInfo("Detected county:#{best[1]}(source #{path} -> #{best[2]})")
        return best[1]
      rescue StandardError => e
        runner.registerWarning("Failed to read OSW while resolving county:#{path};#{e}")
      end
    end
    nil
  end

  def resolve_county_code(runner, explicit_county_code)
    manual = normalize_county_code(explicit_county_code)
    return manual unless manual.nil?

    xml_val = detect_county_from_existing_xml(runner)
    return xml_val unless xml_val.nil?

    env_val = detect_county_from_env(runner)
    return env_val unless env_val.nil?

    osw_val = detect_county_from_osw(runner)
    return osw_val unless osw_val.nil?

    nil
  end


  def ensure_ems_debug_output(workspace, runner)
    objs = workspace.getObjectsByType('Output:EnergyManagementSystem'.to_IddObjectType)

    if objs.empty?
      return add_idf_object(workspace, runner, <<~IDF)
        Output:EnergyManagementSystem,
          Verbose,
          Verbose,
          ErrorsOnly;
      IDF
    end

    objs.first.setString(0, 'Verbose')
    objs.first.setString(1, 'Verbose')
    objs.first.setString(2, 'ErrorsOnly')
    objs.drop(1).each(&:remove)
    true
  end

  def collect_referenced_schedule_objects(workspace, object_specs, exclude_names: [])
    out = []
    exclude_cf = exclude_names.map { |x| cf(x) }

    object_specs.each do |obj_type, field_names|
      workspace.getObjectsByType(obj_type.to_IddObjectType).each do |obj|
        field_names.each do |field_name|
          idx = field_index_by_name_fragment(obj, field_name)
          next if idx.nil?

          sched_name = safe_get_string(obj, idx)
          next if sched_name.nil? || sched_name.empty?
          next if exclude_cf.include?(cf(sched_name))

          sched = find_schedule_exact(workspace, sched_name, ems_actuatable_only: true)
          out << sched unless sched.nil?
        end
      end
    end

    unique_named_objects(out)
  end

  def discover_backup_heating_availability_schedule_objects(workspace)
    out = []

    workspace.getObjectsByType('Coil:Heating:Electric'.to_IddObjectType).each do |obj|
      name_cf = cf(obj_name(obj))
      next unless %w[backup supp supplemental].any? { |x| name_cf.include?(x) }

      idx = field_index_by_name_fragment(obj, 'Availability Schedule Name')
      next if idx.nil?

      sched_name = safe_get_string(obj, idx)
      next if sched_name.nil? || sched_name.empty?
      next if ['always on discrete', 'always on continuous'].include?(cf(sched_name))

      sched = find_schedule_exact(workspace, sched_name, ems_actuatable_only: true)
      out << sched unless sched.nil?
    end

    unique_named_objects(out)
  end


  def default_json_root_dir
    File.expand_path(File.join(File.dirname(__FILE__), 'resources', 'outage_json'))
  end

  def ensure_schedule_type_limits_and_always_on(workspace, runner)
    ok = true

    unless find_object_by_type_and_name(workspace, 'ScheduleTypeLimits', 'Fractional')
      ok &&= add_idf_object(workspace, runner, <<~IDF)
        ScheduleTypeLimits,
          Fractional,
          0,
          1,
          Continuous,
          Dimensionless;
      IDF
    end

    unless find_object_by_type_and_name(workspace, 'ScheduleTypeLimits', 'OnOff')
      ok &&= add_idf_object(workspace, runner, <<~IDF)
        ScheduleTypeLimits,
          OnOff,
          0,
          1,
          Discrete;
      IDF
    end

    if find_schedule_exact(workspace, 'Always On Discrete').nil?
      ok &&= add_idf_object(workspace, runner, <<~IDF)
        Schedule:Constant,
          Always On Discrete,
          OnOff,
          1;
      IDF
    end

    ok
  end

  def ensure_sql_output_enabled(workspace, runner)
    ocf = workspace.getObjectsByType('OutputControl:Files'.to_IddObjectType).first
    if ocf.nil?
      ok = add_idf_object(workspace, runner, <<~IDF)
        OutputControl:Files,
          No,
          No,
          No,
          No,
          No,
          Yes,
          Yes,
          No,
          Yes,
          Yes,
          Yes,
          Yes,
          No,
          No,
          No;
      IDF
      return false unless ok
    else
      ocf.setString(5, 'Yes') if ocf.numFields > 5
    end

    sqlite_objects = workspace.getObjectsByType('Output:SQLite'.to_IddObjectType)
    if sqlite_objects.empty?
      add_idf_object(workspace, runner, <<~IDF)
        Output:SQLite,
          SimpleAndTabular;
      IDF
    else
      sqlite_objects.first.setString(0, 'SimpleAndTabular')
      sqlite_objects.drop(1).each(&:remove)
      true
    end
  end

  def build_constant_schedule(name, type_limits, value)
    <<~IDF
      Schedule:Constant,
        #{name},
        #{type_limits},
        #{value};
    IDF
  end


  def resolve_outage_json_path(json_dir, climate, period)
    candidates = [
      File.join(json_dir, "#{climate}_#{period}_Scenario_EPW_Scenario_EPW.json"),
      File.join(json_dir, "#{climate}_#{period}_scenario_EPW_Scenario_EPW.json"),
      File.join(json_dir, "#{climate}_#{period}.json")
    ]
    candidates.find { |p| File.exist?(p) }
  end

  def generate_grid_schedule(county:, climate:, period:, total_hours:, json_dir:, runner:)
    valid_climates = %w[ssp126 ssp245 ssp585]
    valid_periods = %w[2020s 2030s 2040s 2050s]

    raise "ssp_scenario must be one of #{valid_climates.join(', ')} ." unless valid_climates.include?(climate)
    raise "scenario_year must be one of #{valid_periods.join(', ')} ." unless valid_periods.include?(period)

    json_path = resolve_outage_json_path(json_dir, climate, period)
    unless json_path
      runner.registerWarning(
        "Outage JSON was not found.Tried:#{File.join(json_dir, "#{climate}_#{period}_Scenario_EPW_Scenario_EPW.json")} / " \
        "#{File.join(json_dir, "
      )
      return Array.new(total_hours, 1)
    end

    runner.registerInfo("Using outage JSON:#{json_path}")

    data = JSON.parse(File.read(json_path))
    schedule = Array.new(total_hours, 1)

    unless data.key?(county)
      runner.registerWarning("County was not found in outage JSON=#{county};assuming utility service is available all year.")
      return schedule
    end

    data[county].each do |start_end|
      s = [start_end[0].to_i - 1, 0].max
      e = [start_end[1].to_i, total_hours].min
      (s...e).each { |i| schedule[i] = 0 }
    end

    schedule
  end

  def build_schedule_compact(values, schedule_name, schedule_type_limits: 'Fractional', year: 2018, total_hours: 8760)
    vals = values[0, total_hours] || []
    vals += Array.new(total_hours - vals.size, 0) if vals.size < total_hours

    start_date = Date.new(year, 1, 1)
    days = vals.each_slice(24).to_a

    fields = []
    fields << 'Schedule:Compact'
    fields << "  #{schedule_name}"
    fields << "  #{schedule_type_limits}"

    i = 0
    while i < days.size
      day_pattern = days[i]
      j = i
      j += 1 while (j + 1) < days.size && days[j + 1] == day_pattern

      through_date = start_date + j
      fields << format('  Through: %02d/%02d', through_date.month, through_date.day)
      fields << '  For: AllDays'

      current = day_pattern[0]
      (1..24).each do |hr|
        is_last = (hr == 24)
        nxt = is_last ? nil : day_pattern[hr]
        if is_last || (nxt != current)
          fields << format('  Until: %02d:00, %s', hr, current)
          current = nxt unless is_last
        end
      end

      i = j + 1
    end

    "#{fields.join(",\n")};"
  end


  # RunPeriod / outage diagnostics


  def extract_runperiod_hour_range(workspace, runner)
    rp = workspace.getObjectsByType('RunPeriod'.to_IddObjectType).first
    if rp.nil?
      runner.registerWarning('RunPeriod was not found;using hours 1-8760 for outage calculations.')
      return [1, 8760]
    end

    begin_month = safe_get_string(rp, 1).to_i
    begin_day   = safe_get_string(rp, 2).to_i
    begin_year  = safe_get_string(rp, 3).to_i
    end_month   = safe_get_string(rp, 4).to_i
    end_day     = safe_get_string(rp, 5).to_i
    end_year    = safe_get_string(rp, 6).to_i

    year = begin_year > 0 ? begin_year : 2018
    start_date = Date.new(year, begin_month, begin_day)
    end_date   = Date.new((end_year > 0 ? end_year : year), end_month, end_day)

    year0 = Date.new(year, 1, 1)
    start_hour = ((start_date - year0).to_i * 24) + 1
    end_hour = (((end_date - year0).to_i + 1) * 24)

    runner.registerInfo("RunPeriod hour range:#{start_hour}..#{end_hour} (#{start_date} ~ #{end_date})")
    [start_hour, end_hour]
  rescue StandardError => e
    runner.registerWarning("Failed to parse RunPeriod:#{e};using hours 1-8760 for outage calculations.")
    [1, 8760]
  end

  def count_ones_in_range(arr, start_hour, end_hour)
    s = [start_hour - 1, 0].max
    e = [end_hour - 1, arr.size - 1].min
    return 0 if e < s

    arr[s..e].count(1)
  end

  def apply_debug_forced_outage(outage_schedule, start_hour, duration_hours, runner)
    return outage_schedule if duration_hours.to_i <= 0

    out = outage_schedule.dup
    s = [start_hour.to_i - 1, 0].max
    e = [s + duration_hours.to_i - 1, out.size - 1].min
    (s..e).each { |i| out[i] = 1 }

    runner.registerWarning("Debug mode enabled:forces an outage,start_hour=#{start_hour}, duration_hours=#{duration_hours}")
    out
  end


  def ev_related_text?(text)
    n = cf(text)
    return false if n.empty?

    return true if n.include?('electric_vehicle')
    return true if n.include?('plug_in_vehicle')
    return true if n.include?('vehicle_to_home')
    return true if n.include?('vehicle_to_building')
    return true if n.include?('v2h')
    return true if n.include?('v2b')
    return true if n.include?('vehicle')

    !!(n =~ /(^|[^a-z0-9])ev([^a-z0-9]|$)/)
  end

  def stationary_storage_candidate?(obj)
    type_cf = cf(obj_type_name(obj))
    name_cf = cf(obj_name(obj))

    return false unless type_cf.start_with?('electricloadcenter:storage:')
    return false if type_cf.include?('converter')
    return false if ev_related_text?(name_cf)

    true
  end

  def pv_generator_candidate?(obj)
    type_cf = cf(obj_type_name(obj))
    name_cf = cf(obj_name(obj))

    return true if type_cf.start_with?('generator:photovoltaic')
    return true if type_cf.start_with?('generator:pvwatts')
    return true if name_cf.include?('pv')

    false
  end

  def build_storage_usage_map(workspace)
    usage = Hash.new { |h, k| h[k] = [] }

    workspace.getObjectsByType('ElectricLoadCenter:Distribution'.to_IddObjectType).each do |dist_obj|
      storage_name = distribution_storage_name(dist_obj)
      next if storage_name.nil? || storage_name.empty?

      usage[cf(storage_name)] << obj_name(dist_obj)
    end

    usage
  end

  def storage_used_by_other_distribution?(storage_name, current_distribution_name, storage_usage_map)
    users = Array(storage_usage_map[cf(storage_name)]).map { |n| cf(n) }.uniq
    return false if users.empty?

    current_cf = cf(current_distribution_name)
    users.any? { |dist_name| dist_name != current_cf }
  end

  def storage_unassigned?(storage_name, storage_usage_map)
    Array(storage_usage_map[cf(storage_name)]).empty?
  end

  def storage_candidate_sort_key(obj)
    [
      cf(obj_type_name(obj)).include?('liionnmcbattery') ? 0 : 1,
      cf(obj_name(obj)).include?('battery') ? 0 : 1,
      cf(obj_name(obj))
    ]
  end

  def discover_first_inverter_name(workspace)
    candidates = workspace.objects.select do |obj|
      type_cf = cf(obj_type_name(obj))
      type_cf.start_with?('electricloadcenter:inverter') && !ev_related_text?(obj_name(obj))
    end

    candidates = unique_named_objects(candidates).sort_by { |obj| cf(obj_name(obj)) }
    return nil if candidates.empty?

    obj_name(candidates.first)
  end

  def discover_pv_generator_name(workspace, generator_list_name)
    return nil if generator_list_name.nil? || generator_list_name.empty?

    gl = find_object_by_type_and_name(workspace, 'ElectricLoadCenter:Generators', generator_list_name)
    return nil if gl.nil?

    i = 1
    while i < gl.numFields
      gen_name = safe_get_string(gl, i)
      if !gen_name.nil? && !gen_name.empty?
        gen_obj = find_named_with_type_prefix(workspace, gen_name, ['Generator:'])
        return gen_name if !gen_obj.nil? && pv_generator_candidate?(gen_obj)
      end
      i += 3
    end

    nil
  end

  def pick_primary_distribution(workspace)
    dists = workspace.getObjectsByType('ElectricLoadCenter:Distribution'.to_IddObjectType)
    raise MissingHybridDerError, 'ElectricLoadCenter:Distribution was not found.' if dists.empty?

    dists.each { |obj| return obj if cf(obj_name(obj)) == cf('PVSystem elec load center dist') }
    dists.each { |obj| return obj if cf(obj_name(obj)).include?('pv') || cf(distribution_generator_list_name(obj)).include?('pv') }

    dists[0]
  end


  def discover_hybrid_core(workspace)
    dists = workspace.getObjectsByType('ElectricLoadCenter:Distribution'.to_IddObjectType)
    raise MissingHybridDerError, 'ElectricLoadCenter:Distribution was not found.' if dists.empty?

    storage_usage_map = build_storage_usage_map(workspace)
    fallback_inverter_name = discover_first_inverter_name(workspace)


    candidates = []

    dists.each do |dist_obj|
      distribution_name = obj_name(dist_obj)
      generator_list_name = distribution_generator_list_name(dist_obj)
      next if generator_list_name.nil? || generator_list_name.empty?

      pv_generator_name = discover_pv_generator_name(workspace, generator_list_name)
      next if pv_generator_name.nil? || pv_generator_name.empty?

      battery_name = distribution_storage_name(dist_obj)
      next if battery_name.nil? || battery_name.empty?

      battery_obj = find_named_with_type_prefix(workspace, battery_name, ['ElectricLoadCenter:Storage:'])
      next if battery_obj.nil?
      next unless stationary_storage_candidate?(battery_obj)
      next if storage_used_by_other_distribution?(obj_name(battery_obj), distribution_name, storage_usage_map)

      inverter_name = distribution_inverter_name(dist_obj)
      inverter_name = fallback_inverter_name if inverter_name.nil? || inverter_name.empty?
      next if inverter_name.nil? || inverter_name.empty?
      next if ev_related_text?(distribution_name)
      next if ev_related_text?(obj_name(battery_obj))

      candidates << {
        distribution_name: distribution_name,
        distribution_obj_type: obj_type_name(dist_obj),
        generator_list_name: generator_list_name,
        inverter_name: inverter_name,
        battery_name: obj_name(battery_obj),
        battery_obj_type: obj_type_name(battery_obj),
        pv_generator_name: pv_generator_name
      }
    end

    unless candidates.empty?
      candidates.sort_by! do |c|
        [
          cf(c[:distribution_name]) == cf('PVSystem elec load center dist') ? 0 : 1,
          cf(c[:battery_obj_type]).include?('liionnmcbattery') ? 0 : 1,
          cf(c[:distribution_name])
        ]
      end
      return candidates.first
    end


    dist_obj = pick_primary_distribution(workspace)
    distribution_name = obj_name(dist_obj)
    generator_list_name = distribution_generator_list_name(dist_obj)

    raise MissingHybridDerError, "Distribution=#{distribution_name} is missing Generator List Name." if generator_list_name.nil? || generator_list_name.empty?

    pv_generator_name = discover_pv_generator_name(workspace, generator_list_name)
    raise MissingHybridDerError, 'No PV generator object was found.' if pv_generator_name.nil? || pv_generator_name.empty?

    inverter_name = distribution_inverter_name(dist_obj)
    inverter_name = fallback_inverter_name if inverter_name.nil? || inverter_name.empty?
    raise MissingHybridDerError, 'Inverter was not found.' if inverter_name.nil? || inverter_name.empty?

    battery_obj = nil
    battery_name = distribution_storage_name(dist_obj)

    unless battery_name.nil? || battery_name.empty?
      preferred_obj = find_named_with_type_prefix(workspace, battery_name, ['ElectricLoadCenter:Storage:'])
      if !preferred_obj.nil? &&
         stationary_storage_candidate?(preferred_obj) &&
         !storage_used_by_other_distribution?(obj_name(preferred_obj), distribution_name, storage_usage_map)
        battery_obj = preferred_obj
      end
    end

    if battery_obj.nil?
      storage_candidates = workspace.objects.select do |obj|
        stationary_storage_candidate?(obj) && storage_unassigned?(obj_name(obj), storage_usage_map)
      end

      storage_candidates = unique_named_objects(storage_candidates).sort_by { |obj| storage_candidate_sort_key(obj) }
      battery_obj = storage_candidates.first
    end

    raise MissingHybridDerError, 'No eligible stationary battery was found(EVs, converters, and storage assigned to another distribution are excluded).' if battery_obj.nil?

    {
      distribution_name: distribution_name,
      distribution_obj_type: obj_type_name(dist_obj),
      generator_list_name: generator_list_name,
      inverter_name: inverter_name,
      battery_name: obj_name(battery_obj),
      battery_obj_type: obj_type_name(battery_obj),
      pv_generator_name: pv_generator_name
    }
  end


  def find_schedule_exact(workspace, schedule_name, ems_actuatable_only: false)
    workspace.objects.each do |obj|
      next unless schedule_object?(obj)
      next unless cf(obj_name(obj)) == cf(schedule_name)
      next if ems_actuatable_only && !ems_actuatable_schedule_type?(obj)

      return obj
    end
    nil
  end

  def find_schedules_by_patterns(workspace, patterns, ems_actuatable_only: false)
    out = []
    workspace.objects.each do |obj|
      next unless schedule_object?(obj)
      next if ems_actuatable_only && !ems_actuatable_schedule_type?(obj)

      n = cf(obj_name(obj))
      out << obj if patterns.any? { |p| n.include?(cf(p)) }
    end
    unique_named_objects(out)
  end

  def fridge_equipment_name?(name)
    n = cf(name)
    n.include?('fridge') || n.include?('refrigerator')
  end

  def build_object_actuators(objects, component_type:, control_type:, prefix:)
    specs = []
    unique_named_objects(objects).each_with_index do |obj, i|
      specs << {
        ems_name: make_ems_name(prefix, obj_name(obj), i + 1),
        component_name: obj_name(obj),
        component_type: component_type,
        control_type: control_type
      }
    end
    specs
  end

  def build_schedule_actuators(schedules, prefix:)
    specs = []
    unique_named_objects(schedules).each_with_index do |obj, i|
      next unless ems_actuatable_schedule_type?(obj)

      specs << {
        ems_name: make_ems_name(prefix, obj_name(obj), i + 1),
        component_name: obj_name(obj),
        component_type: obj_type_name(obj),
        control_type: 'Schedule Value'
      }
    end
    specs
  end

  def discover_optional_controls(workspace)
    controls = empty_optional_controls

    electric_eq = unique_named_objects(workspace.getObjectsByType('ElectricEquipment'.to_IddObjectType))
    lights = unique_named_objects(workspace.getObjectsByType('Lights'.to_IddObjectType))

    fridge_eq_objs = electric_eq.select { |o| fridge_equipment_name?(obj_name(o)) }
    noncritical_eq_objs = electric_eq.reject { |o| fridge_equipment_name?(obj_name(o)) }

    bath_scheds = find_schedules_by_patterns(workspace, ['mech vent bath fan', 'bath fan'], ems_actuatable_only: true)
    range_scheds = find_schedules_by_patterns(workspace, ['mech vent range fan', 'range fan'], ems_actuatable_only: true)
    wh_scheds = find_schedules_by_patterns(workspace, ['water heater setpoint'], ems_actuatable_only: true)

    hvac_fraction_scheds = find_schedules_by_patterns(workspace, ['sequential fraction schedule'], ems_actuatable_only: true)
    heating_sp_scheds = find_schedules_by_patterns(workspace, ['heating setpoint'], ems_actuatable_only: true)
    cooling_sp_scheds = find_schedules_by_patterns(workspace, ['cooling setpoint'], ems_actuatable_only: true)

    ext_light_scheds = collect_referenced_schedule_objects(
      workspace,
      [
        ['Exterior:Lights', ['Schedule Name']]
      ]
    )

    hvac_avail_sched_objs = collect_referenced_schedule_objects(
      workspace,
      [
        ['AvailabilityManager:Scheduled', ['Schedule Name']],
        ['AirTerminal:SingleDuct:ConstantVolume:NoReheat', ['Availability Schedule Name']],
        ['AirLoopHVAC:UnitarySystem', ['Availability Schedule Name']],
        ['ZoneHVAC:IdealLoadsAirSystem', ['Availability Schedule Name', 'Heating Availability Schedule Name', 'Cooling Availability Schedule Name']],
        ['ZoneHVAC:Baseboard:Convective:Water', ['Availability Schedule Name']],
        ['Fan:SystemModel', ['Availability Schedule Name']],
        ['Coil:Cooling:DX:SingleSpeed', ['Availability Schedule Name']]
      ],
      exclude_names: [
        'Always On Discrete',
        'Always On Continuous',
        ISLAND_HVAC_SCHEDULE_NAME,
        BACKUP_HTG_SCHEDULE_NAME
      ]
    )

    backup_htg_avail_sched_objs = discover_backup_heating_availability_schedule_objects(workspace)

    controls[:noncritical_eq] = build_object_actuators(
      noncritical_eq_objs,
      component_type: 'ElectricEquipment',
      control_type: 'Electricity Rate',
      prefix: 'nc_eq'
    )

    controls[:fridge_eq] = build_object_actuators(
      fridge_eq_objs,
      component_type: 'ElectricEquipment',
      control_type: 'Electricity Rate',
      prefix: 'fridge_eq'
    )

    controls[:interior_lights] = build_object_actuators(
      lights,
      component_type: 'Lights',
      control_type: 'Electricity Rate',
      prefix: 'int_light'
    )

    controls[:exterior_lights] = build_schedule_actuators(
      ext_light_scheds,
      prefix: 'ext_light'
    )

    controls[:bath_fan_schedules] = build_schedule_actuators(
      bath_scheds,
      prefix: 'bath_sched'
    )

    controls[:range_fan_schedules] = build_schedule_actuators(
      range_scheds,
      prefix: 'range_sched'
    )

    controls[:water_heater_setpoints] = build_schedule_actuators(
      wh_scheds,
      prefix: 'wh_sp'
    )

    controls[:hvac_fraction_schedules] = build_schedule_actuators(
      hvac_fraction_scheds,
      prefix: 'hvac_frac'
    )

    controls[:heating_setpoint_schedules] = build_schedule_actuators(
      heating_sp_scheds,
      prefix: 'htg_sp'
    )

    controls[:cooling_setpoint_schedules] = build_schedule_actuators(
      cooling_sp_scheds,
      prefix: 'clg_sp'
    )

    controls[:hvac_availability_schedules] = build_schedule_actuators(
      hvac_avail_sched_objs,
      prefix: 'hvac_avail'
    )

    controls[:backup_htg_availability_schedules] = build_schedule_actuators(
      backup_htg_avail_sched_objs,
      prefix: 'backup_avail'
    )

    controls
  end

  def all_optional_specs(controls)
    controls = normalize_optional_controls(controls)

    keys = [
      :noncritical_eq,
      :fridge_eq,
      :interior_lights,
      :exterior_lights,
      :bath_fan_schedules,
      :range_fan_schedules,
      :water_heater_setpoints,
      :hvac_fraction_schedules,
      :heating_setpoint_schedules,
      :cooling_setpoint_schedules,
      :hvac_availability_schedules,
      :backup_htg_availability_schedules
    ]

    out = []
    seen = {}

    keys.each do |k|
      Array(controls[k]).each do |spec|
        next if spec.nil?

        key = [
          cf(spec[:ems_name]),
          cf(spec[:component_name]),
          cf(spec[:component_type]),
          cf(spec[:control_type])
        ]
        next if seen[key]

        seen[key] = true
        out << spec
      end
    end

    out
  end


  # patch battery / distribution / hvac


  def patch_battery_initial_soc(workspace, runner, core, initial_soc)
    battery_obj = find_named_with_type_prefix(workspace, core[:battery_name], ['ElectricLoadCenter:Storage:'])
    return if battery_obj.nil?

    unless cf(core[:battery_obj_type]) == cf('ElectricLoadCenter:Storage:LiIonNMCBattery')
      runner.registerWarning("battery  #{core[:battery_obj_type]},is not LiIonNMCBattery;skipping the initial SOC patch.")
      return
    end

    idx = field_index_by_name_fragment(battery_obj, 'Initial Fractional State of Charge')
    if idx.nil?
      runner.registerWarning("Battery was not found=#{core[:battery_name]} Initial Fractional State of Charge field was not found.")
      return
    end

    battery_obj.setString(idx, format('%.3f', initial_soc))
  end

  def patch_distribution_object(workspace, core, battery_min_soc:, battery_max_soc:, charge_power_w:, discharge_power_w:, runner:)
    dist = find_object_by_type_and_name(workspace, 'ElectricLoadCenter:Distribution', core[:distribution_name])
    if dist.nil?
      runner.registerError("ElectricLoadCenter:Distribution was not found=#{core[:distribution_name]}")
      return false
    end

    begin
      set_field_value_by_name_fragments(dist, ['Electrical Buss Type'], 'DirectCurrentWithInverterACStorage')
      set_field_value_by_name_fragments(dist, ['Inverter Name', 'Inverter Object Name'], core[:inverter_name], required: false)
      set_field_value_by_name_fragments(dist, ['Electrical Storage Object Name'], core[:battery_name])
      set_field_value_by_name_fragments(dist, ['Storage Operation Scheme'], 'TrackChargeDischargeSchedules')

      idx = field_index_by_name_fragments(dist, ['Storage Control Track Meter Name'])
      dist.setString(idx, '') unless idx.nil?

      set_field_value_by_name_fragments(dist, ['Storage Converter Object Name', 'Converter Object Name'], CONVERTER_NAME)
      set_field_value_by_name_fragments(dist, ['Maximum Storage State of Charge Fraction'], format('%.3f', battery_max_soc))
      set_field_value_by_name_fragments(dist, ['Minimum Storage State of Charge Fraction'], format('%.3f', battery_min_soc))
      set_field_value_by_name_fragments(dist, ['Design Storage Control Charge Power'], format('%.1f', charge_power_w))
      set_field_value_by_name_fragments(dist, ['Storage Charge Power Fraction Schedule Name', 'Charge Power Fraction Schedule Name'], CHARGE_SCHEDULE_NAME)
      set_field_value_by_name_fragments(dist, ['Design Storage Control Discharge Power'], format('%.1f', discharge_power_w))
      set_field_value_by_name_fragments(dist, ['Storage Discharge Power Fraction Schedule Name', 'Discharge Power Fraction Schedule Name'], DISCHARGE_SCHEDULE_NAME)
    rescue StandardError => e
      runner.registerWarning("Distribution field patch failed; rebuilding the object:#{e}")

      remove_named_object(workspace, 'ElectricLoadCenter:Distribution', core[:distribution_name])

      return add_idf_object(
        workspace,
        runner,
        build_distribution_idf(
          core,
          battery_min_soc: battery_min_soc,
          battery_max_soc: battery_max_soc,
          charge_power_w: charge_power_w,
          discharge_power_w: discharge_power_w
        )
      )
    end

    dist_text = dist.to_s
    expected_tokens = [
      'TrackChargeDischargeSchedules',
      CONVERTER_NAME,
      CHARGE_SCHEDULE_NAME,
      DISCHARGE_SCHEDULE_NAME
    ]

    unless expected_tokens.all? { |tok| cf(dist_text).include?(cf(tok)) }
      runner.registerWarning('Distribution field patch validation failed; rebuilding the object.')

      remove_named_object(workspace, 'ElectricLoadCenter:Distribution', core[:distribution_name])

      return add_idf_object(
        workspace,
        runner,
        build_distribution_idf(
          core,
          battery_min_soc: battery_min_soc,
          battery_max_soc: battery_max_soc,
          charge_power_w: charge_power_w,
          discharge_power_w: discharge_power_w
        )
      )
    end

    true
  end

  def patch_if_blank_or_always_on(obj, idx, new_schedule_name)
    current = safe_get_string(obj, idx)
    return false unless current.nil? || current.empty? || ['always on discrete', 'always on continuous'].include?(cf(current))

    obj.setString(idx, new_schedule_name)
    true
  end

  def patch_hvac_availability_objects(workspace, runner)
    patched = 0

    [
      ['AvailabilityManager:Scheduled', ['Schedule Name'], ISLAND_HVAC_SCHEDULE_NAME],
      ['AirTerminal:SingleDuct:ConstantVolume:NoReheat', ['Availability Schedule Name'], ISLAND_HVAC_SCHEDULE_NAME],
      ['AirLoopHVAC:UnitarySystem', ['Availability Schedule Name'], ISLAND_HVAC_SCHEDULE_NAME],
      ['ZoneHVAC:IdealLoadsAirSystem', ['Availability Schedule Name', 'Heating Availability Schedule Name', 'Cooling Availability Schedule Name'], ISLAND_HVAC_SCHEDULE_NAME],
      ['ZoneHVAC:Baseboard:Convective:Water', ['Availability Schedule Name'], ISLAND_HVAC_SCHEDULE_NAME],
      ['Fan:SystemModel', ['Availability Schedule Name'], ISLAND_HVAC_SCHEDULE_NAME],
      ['Coil:Cooling:DX:SingleSpeed', ['Availability Schedule Name'], ISLAND_HVAC_SCHEDULE_NAME]
    ].each do |obj_type, field_names, target_schedule|
      workspace.getObjectsByType(obj_type.to_IddObjectType).each do |obj|
        field_names.each do |field_name|
          idx = field_index_by_name_fragment(obj, field_name)
          next if idx.nil?

          patched += 1 if patch_if_blank_or_always_on(obj, idx, target_schedule)
        end
      end
    end

    workspace.getObjectsByType('Coil:Heating:Electric'.to_IddObjectType).each do |obj|
      name_cf = cf(obj_name(obj))
      next unless %w[backup supp supplemental].any? { |x| name_cf.include?(x) }

      idx = field_index_by_name_fragment(obj, 'Availability Schedule Name')
      next if idx.nil?

      patched += 1 if patch_if_blank_or_always_on(obj, idx, BACKUP_HTG_SCHEDULE_NAME)
    end

    workspace.getObjectsByType('PlantEquipmentOperationSchemes'.to_IddObjectType).each do |obj|
      idx = 3
      while idx < obj.numFields
        patched += 1 if patch_if_blank_or_always_on(obj, idx, ISLAND_HVAC_SCHEDULE_NAME)
        idx += 3
      end
    end

    runner.registerInfo("HVAC availability patches:#{patched}")
  end


  def remove_previous_hybrid_patch(workspace, core = nil)
    [
      ISLAND_SCHEDULE_NAME,
      CHARGE_SCHEDULE_NAME,
      DISCHARGE_SCHEDULE_NAME,
      ISLAND_HVAC_SCHEDULE_NAME,
      BACKUP_HTG_SCHEDULE_NAME
    ].each do |sched_name|
      remove_any_named_schedule(workspace, sched_name)
    end

    remove_named_object(workspace, 'ElectricLoadCenter:Storage:Converter', CONVERTER_NAME)

    [
      'EnergyManagementSystem:GlobalVariable',
      'EnergyManagementSystem:Sensor',
      'EnergyManagementSystem:Actuator',
      'EnergyManagementSystem:Program',
      'EnergyManagementSystem:ProgramCallingManager',
      'EnergyManagementSystem:OutputVariable'
    ].each do |obj_type|
      remove_ems_objects_with_prefix(workspace, obj_type, PATCH_PREFIX)
    end

    remove_named_object(workspace, 'EnergyManagementSystem:Program', PROGRAM_NAME)
    remove_named_object(workspace, 'EnergyManagementSystem:ProgramCallingManager', PCM_NAME)

    remove_output_variable(workspace, ISLAND_SCHEDULE_NAME, 'Schedule Value')
    remove_output_variable(workspace, CHARGE_SCHEDULE_NAME, 'Schedule Value')
    remove_output_variable(workspace, DISCHARGE_SCHEDULE_NAME, 'Schedule Value')
    remove_output_variable(workspace, ISLAND_HVAC_SCHEDULE_NAME, 'Schedule Value')
    remove_output_variable(workspace, BACKUP_HTG_SCHEDULE_NAME, 'Schedule Value')

    remove_output_meter(workspace, 'ElectricityPurchased:Facility')
    remove_output_meter(workspace, 'Electricity:Facility')

    return if core.nil?

    remove_output_variable(workspace, core[:battery_name], 'Electric Storage Charge Fraction')
    remove_output_variable(workspace, core[:battery_name], 'Electric Storage Charge Energy')
    remove_output_variable(workspace, core[:battery_name], 'Electric Storage Discharge Energy')
    remove_output_variable(workspace, core[:battery_name], 'Electric Storage Energy')

    return if core[:pv_generator_name].nil?

    remove_output_variable(workspace, core[:pv_generator_name], 'Generator Produced DC Electricity Rate')
  end


  # EMS render helpers


  def render_ems_global_variables(var_names)
    var_names.map do |name|
      <<~IDF
        EnergyManagementSystem:GlobalVariable,
          #{name};
      IDF
    end
  end

  def render_ems_actuators(specs)
    specs.map do |spec|
      <<~IDF
        EnergyManagementSystem:Actuator,
          #{spec[:ems_name]},
          #{spec[:component_name]},
          #{spec[:component_type]},
          #{spec[:control_type]};
      IDF
    end
  end

  def set_lines(specs, value)
    Array(specs).map { |spec| "Set #{spec[:ems_name]} = #{value}" }
  end

  def indent(lines, level)
    prefix = '  ' * level
    lines.map { |line| "#{prefix}#{line}" }
  end


  def build_program_lines(
    controls,
    has_pv_sensor:,
    battery_initial_soc:,
    battery_min_soc:,
    battery_max_soc:,
    battery_charge_power_w:,
    battery_discharge_power_w:,
    critical_misc_w:,
    critical_interior_lighting_w:,
    critical_hvac_w:
  )
    controls = normalize_optional_controls(controls)

    lines = []

    if has_pv_sensor
      lines << "Set #{PATCH_PREFIX}pv_ac_avail = #{PATCH_PREFIX}pv_dc_rate_s * 0.96"
    else
      lines << "Set #{PATCH_PREFIX}pv_ac_avail = 0.0"
    end

    lines += [
      "Set #{PATCH_PREFIX}critical_target_w = 0.0",
      "Set #{PATCH_PREFIX}charge_frac_cmd = 0.0",
      "Set #{PATCH_PREFIX}discharge_frac_cmd = 0.0",
      "Set #{PATCH_PREFIX}hvac_allow = 0.0",
      "If #{PATCH_PREFIX}mode_s > 0",
      "  Set #{PATCH_PREFIX}island_hvac_avail_act = 0",
      "  Set #{PATCH_PREFIX}backup_htg_avail_act = 0"
    ]

    island_defaults = []
    island_defaults += set_lines(controls[:noncritical_eq], '0.0')
    island_defaults += set_lines(controls[:exterior_lights], '0.0')
    island_defaults += set_lines(controls[:bath_fan_schedules], '0.0')
    island_defaults += set_lines(controls[:range_fan_schedules], '0.0')
    island_defaults += set_lines(controls[:water_heater_setpoints], '15.0')
    island_defaults += set_lines(controls[:heating_setpoint_schedules], 'Null')
    island_defaults += set_lines(controls[:cooling_setpoint_schedules], 'Null')
    island_defaults += set_lines(controls[:hvac_fraction_schedules], '0.0')
    island_defaults += set_lines(controls[:interior_lights], '0.0')
    island_defaults += set_lines(controls[:hvac_availability_schedules], '0.0')
    island_defaults += set_lines(controls[:backup_htg_availability_schedules], '0.0')
    lines += indent(island_defaults, 1)

    if controls[:interior_lights].any?
      per_light_w = critical_interior_lighting_w / controls[:interior_lights].size.to_f
      lines << "  If (#{PATCH_PREFIX}battery_soc_s > 0.15) || (#{PATCH_PREFIX}pv_ac_avail > 250.0)"
      lines += indent(set_lines(controls[:interior_lights], format('%.6f', per_light_w)), 2)
      lines << '  Else'
      lines += indent(set_lines(controls[:interior_lights], '0.0'), 2)
      lines << '  EndIf'
    end

    if controls[:fridge_eq].any?
      lines << "  If (#{PATCH_PREFIX}battery_soc_s <= 0.08) && (#{PATCH_PREFIX}pv_ac_avail < 120.0)"
      lines += indent(set_lines(controls[:fridge_eq], '0.0'), 2)
      lines << '  Else'
      lines += indent(set_lines(controls[:fridge_eq], 'Null'), 2)
      lines << '  EndIf'
    end

    lines << "  If (#{PATCH_PREFIX}battery_soc_s > 0.25) || (#{PATCH_PREFIX}pv_ac_avail > 1200.0)"
    lines << "    Set #{PATCH_PREFIX}island_hvac_avail_act = 1"
    lines << "    Set #{PATCH_PREFIX}backup_htg_avail_act = 1"
    lines << "    Set #{PATCH_PREFIX}hvac_allow = 1"
    lines += indent(set_lines(controls[:hvac_fraction_schedules], '1.0'), 2)
    lines += indent(set_lines(controls[:hvac_availability_schedules], 'Null'), 2)
    lines += indent(set_lines(controls[:backup_htg_availability_schedules], 'Null'), 2)
    lines << '  EndIf'

    lines << format("  Set #{PATCH_PREFIX}critical_target_w = %.1f", critical_misc_w)
    if controls[:interior_lights].any?
      lines << format("  Set #{PATCH_PREFIX}critical_target_w = #{PATCH_PREFIX}critical_target_w + %.1f", critical_interior_lighting_w)
    end
    lines << "  If #{PATCH_PREFIX}hvac_allow > 0"
    lines << format("    Set #{PATCH_PREFIX}critical_target_w = #{PATCH_PREFIX}critical_target_w + %.1f", critical_hvac_w)
    lines << '  EndIf'

    lines += [
      "  Set net_storage_need_w = #{PATCH_PREFIX}critical_target_w - #{PATCH_PREFIX}pv_ac_avail",
      '  If net_storage_need_w > 0',
      "    Set #{PATCH_PREFIX}charge_frac_cmd = 0.0",
      "    Set #{PATCH_PREFIX}discharge_frac_cmd = net_storage_need_w / #{format('%.6f', battery_discharge_power_w)}",
      "    Set #{PATCH_PREFIX}discharge_frac_cmd = @Min #{PATCH_PREFIX}discharge_frac_cmd 1.0",
      "    Set #{PATCH_PREFIX}discharge_frac_cmd = @Max #{PATCH_PREFIX}discharge_frac_cmd 0.0",
      "    Set #{PATCH_PREFIX}battery_charge_sched_act = 0.0",
      format("    If #{PATCH_PREFIX}battery_soc_s > %.3f", battery_min_soc),
      "      Set #{PATCH_PREFIX}battery_discharge_sched_act = #{PATCH_PREFIX}discharge_frac_cmd",
      '    Else',
      "      Set #{PATCH_PREFIX}battery_discharge_sched_act = 0.0",
      '    EndIf',
      '  Else',
      "    Set #{PATCH_PREFIX}discharge_frac_cmd = 0.0",
      "    Set #{PATCH_PREFIX}battery_discharge_sched_act = 0.0",
      format("    If #{PATCH_PREFIX}battery_soc_s < %.3f", battery_max_soc),
      "      Set #{PATCH_PREFIX}charge_frac_cmd = (0.0 - net_storage_need_w) / #{format('%.6f', battery_charge_power_w)}",
      "      Set #{PATCH_PREFIX}charge_frac_cmd = @Min #{PATCH_PREFIX}charge_frac_cmd 1.0",
      "      Set #{PATCH_PREFIX}charge_frac_cmd = @Max #{PATCH_PREFIX}charge_frac_cmd 0.0",
      "      Set #{PATCH_PREFIX}battery_charge_sched_act = #{PATCH_PREFIX}charge_frac_cmd",
      '    Else',
      "      Set #{PATCH_PREFIX}charge_frac_cmd = 0.0",
      "      Set #{PATCH_PREFIX}battery_charge_sched_act = 0.0",
      '    EndIf',
      '  EndIf',
      'Else',
      "  Set #{PATCH_PREFIX}island_hvac_avail_act = 1",
      "  Set #{PATCH_PREFIX}backup_htg_avail_act = 1"
    ]

    restore = []
    restore += set_lines(controls[:noncritical_eq], 'Null')
    restore += set_lines(controls[:fridge_eq], 'Null')
    restore += set_lines(controls[:interior_lights], 'Null')
    restore += set_lines(controls[:exterior_lights], 'Null')
    restore += set_lines(controls[:bath_fan_schedules], 'Null')
    restore += set_lines(controls[:range_fan_schedules], 'Null')
    restore += set_lines(controls[:water_heater_setpoints], 'Null')
    restore += set_lines(controls[:heating_setpoint_schedules], 'Null')
    restore += set_lines(controls[:cooling_setpoint_schedules], 'Null')
    restore += set_lines(controls[:hvac_fraction_schedules], 'Null')
    restore += set_lines(controls[:hvac_availability_schedules], 'Null')
    restore += set_lines(controls[:backup_htg_availability_schedules], 'Null')
    lines += indent(restore, 1)

    lines += [
      "  Set #{PATCH_PREFIX}battery_discharge_sched_act = 0.0",
      "  Set #{PATCH_PREFIX}discharge_frac_cmd = 0.0",
      format("  If #{PATCH_PREFIX}battery_soc_s < %.3f", battery_initial_soc),
      "    Set #{PATCH_PREFIX}battery_charge_sched_act = 1.0",
      "    Set #{PATCH_PREFIX}charge_frac_cmd = 1.0",
      '  Else',
      "    Set #{PATCH_PREFIX}battery_charge_sched_act = 0.0",
      "    Set #{PATCH_PREFIX}charge_frac_cmd = 0.0",
      '  EndIf',
      'EndIf'
    ]

    lines
  end

  def render_ems_program(name, lines)
    body = lines.each_with_index.map do |line, i|
      suffix = (i == lines.length - 1) ? ';' : ','
      "  #{line}#{suffix}"
    end.join("\n")

    <<~IDF
      EnergyManagementSystem:Program,
        #{name},
    #{body}
    IDF
  end


  def build_hybrid_patch_objects(core, controls, args_hash, island_schedule_idf)
    has_pv_sensor = !core[:pv_generator_name].nil?
    optional_specs = all_optional_specs(controls)

    globals = [
      "#{PATCH_PREFIX}pv_ac_avail",
      "#{PATCH_PREFIX}critical_target_w",
      "#{PATCH_PREFIX}charge_frac_cmd",
      "#{PATCH_PREFIX}discharge_frac_cmd",
      "#{PATCH_PREFIX}hvac_allow"
    ]

    program_lines = build_program_lines(
      controls,
      has_pv_sensor: has_pv_sensor,
      battery_initial_soc: args_hash[:battery_initial_soc],
      battery_min_soc: args_hash[:battery_min_soc],
      battery_max_soc: args_hash[:battery_max_soc],
      battery_charge_power_w: args_hash[:battery_charge_power_w],
      battery_discharge_power_w: args_hash[:battery_discharge_power_w],
      critical_misc_w: args_hash[:critical_misc_w],
      critical_interior_lighting_w: args_hash[:critical_interior_lighting_w],
      critical_hvac_w: args_hash[:critical_hvac_w]
    )

    objs = []
    objs << island_schedule_idf
    objs << build_constant_schedule(CHARGE_SCHEDULE_NAME, 'Fractional', 0)
    objs << build_constant_schedule(DISCHARGE_SCHEDULE_NAME, 'Fractional', 0)
    objs << build_constant_schedule(ISLAND_HVAC_SCHEDULE_NAME, 'OnOff', 1)
    objs << build_constant_schedule(BACKUP_HTG_SCHEDULE_NAME, 'OnOff', 1)

    converter_power_w = [args_hash[:battery_charge_power_w], args_hash[:battery_discharge_power_w]].max
    objs << build_storage_converter_idf(converter_power_w)

    objs += render_ems_global_variables(globals)

    objs << <<~IDF
      EnergyManagementSystem:Sensor,
        #{PATCH_PREFIX}mode_s,
        #{ISLAND_SCHEDULE_NAME},
        Schedule Value;
    IDF

    objs << <<~IDF
      EnergyManagementSystem:Sensor,
        #{PATCH_PREFIX}battery_soc_s,
        #{core[:battery_name]},
        Electric Storage Charge Fraction;
    IDF

    if has_pv_sensor
      objs << <<~IDF
        EnergyManagementSystem:Sensor,
          #{PATCH_PREFIX}pv_dc_rate_s,
          #{core[:pv_generator_name]},
          Generator Produced DC Electricity Rate;
      IDF
    end

    objs << <<~IDF
      EnergyManagementSystem:Actuator,
        #{PATCH_PREFIX}battery_charge_sched_act,
        #{CHARGE_SCHEDULE_NAME},
        Schedule:Constant,
        Schedule Value;
    IDF

    objs << <<~IDF
      EnergyManagementSystem:Actuator,
        #{PATCH_PREFIX}battery_discharge_sched_act,
        #{DISCHARGE_SCHEDULE_NAME},
        Schedule:Constant,
        Schedule Value;
    IDF

    objs << <<~IDF
      EnergyManagementSystem:Actuator,
        #{PATCH_PREFIX}island_hvac_avail_act,
        #{ISLAND_HVAC_SCHEDULE_NAME},
        Schedule:Constant,
        Schedule Value;
    IDF

    objs << <<~IDF
      EnergyManagementSystem:Actuator,
        #{PATCH_PREFIX}backup_htg_avail_act,
        #{BACKUP_HTG_SCHEDULE_NAME},
        Schedule:Constant,
        Schedule Value;
    IDF

    objs += render_ems_actuators(optional_specs)
    objs << render_ems_program(PROGRAM_NAME, program_lines)

    objs << <<~IDF
      EnergyManagementSystem:ProgramCallingManager,
        #{PCM_NAME},
        BeginTimestepBeforePredictor,
        #{PROGRAM_NAME};
    IDF

    objs << <<~IDF
      EnergyManagementSystem:OutputVariable,
        #{PATCH_PREFIX}pv_ac_available,
        #{PATCH_PREFIX}pv_ac_avail,
        Averaged,
        ZoneTimestep,
        #{PROGRAM_NAME},
        W;
    IDF

    objs << <<~IDF
      EnergyManagementSystem:OutputVariable,
        #{PATCH_PREFIX}critical_target_power,
        #{PATCH_PREFIX}critical_target_w,
        Averaged,
        ZoneTimestep,
        #{PROGRAM_NAME},
        W;
    IDF

    objs << <<~IDF
      EnergyManagementSystem:OutputVariable,
        #{PATCH_PREFIX}battery_charge_cmd,
        #{PATCH_PREFIX}charge_frac_cmd,
        Averaged,
        ZoneTimestep,
        #{PROGRAM_NAME},
        None;
    IDF

    objs << <<~IDF
      EnergyManagementSystem:OutputVariable,
        #{PATCH_PREFIX}battery_discharge_cmd,
        #{PATCH_PREFIX}discharge_frac_cmd,
        Averaged,
        ZoneTimestep,
        #{PROGRAM_NAME},
        None;
    IDF

    objs << <<~IDF
      Output:Variable,
        #{ISLAND_SCHEDULE_NAME},
        Schedule Value,
        hourly;
    IDF

    objs << <<~IDF
      Output:Variable,
        #{CHARGE_SCHEDULE_NAME},
        Schedule Value,
        hourly;
    IDF

    objs << <<~IDF
      Output:Variable,
        #{DISCHARGE_SCHEDULE_NAME},
        Schedule Value,
        hourly;
    IDF

    objs << <<~IDF
      Output:Variable,
        #{ISLAND_HVAC_SCHEDULE_NAME},
        Schedule Value,
        hourly;
    IDF

    objs << <<~IDF
      Output:Variable,
        #{BACKUP_HTG_SCHEDULE_NAME},
        Schedule Value,
        hourly;
    IDF

    objs << <<~IDF
      Output:Variable,
        #{core[:battery_name]},
        Electric Storage Charge Fraction,
        hourly;
    IDF

    objs << <<~IDF
      Output:Variable,
        #{core[:battery_name]},
        Electric Storage Charge Energy,
        hourly;
    IDF

    objs << <<~IDF
      Output:Variable,
        #{core[:battery_name]},
        Electric Storage Discharge Energy,
        hourly;
    IDF

    objs << <<~IDF
      Output:Variable,
        #{core[:battery_name]},
        Electric Storage Energy,
        hourly;
    IDF

    if has_pv_sensor
      objs << <<~IDF
        Output:Variable,
          #{core[:pv_generator_name]},
          Generator Produced DC Electricity Rate,
          hourly;
      IDF
    end

    objs << <<~IDF
      Output:Meter,
        ElectricityPurchased:Facility,
        hourly;
    IDF

    objs << <<~IDF
      Output:Meter,
        Electricity:Facility,
        hourly;
    IDF

    objs
  end


  def log_object_field_names(obj, runner, title = nil)
    runner.registerInfo("---- #{title || "#{obj_type_name(obj)} / #{obj_name(obj)}"} fields begin ----")
    i = 0
    loop do
      idd_field = obj.iddObject.getField(i)
      break unless idd_field.is_initialized

      fname = idd_field.get.name.to_s
      val = safe_get_string(obj, i)
      runner.registerInfo("Field #{i}: #{fname} = #{val.nil? ? '<nil>' : val}")
      i += 1
    end
    runner.registerInfo('---- fields end ----')
  end

  def verify_hybrid_patch(workspace, runner, core)
    missing = []

    [
      ISLAND_SCHEDULE_NAME,
      CHARGE_SCHEDULE_NAME,
      DISCHARGE_SCHEDULE_NAME,
      ISLAND_HVAC_SCHEDULE_NAME,
      BACKUP_HTG_SCHEDULE_NAME
    ].each do |sched_name|
      missing << "Missing schedule: #{sched_name}" if find_schedule_exact(workspace, sched_name).nil?
    end

    converter = find_object_by_type_and_name(workspace, 'ElectricLoadCenter:Storage:Converter', CONVERTER_NAME)
    missing << "Missing ElectricLoadCenter:Storage:Converter=#{CONVERTER_NAME}" if converter.nil?

    dist = find_object_by_type_and_name(workspace, 'ElectricLoadCenter:Distribution', core[:distribution_name])
    if dist.nil?
      missing << "Missing ElectricLoadCenter:Distribution=#{core[:distribution_name]}"
    else
      dist_text = dist.to_s
      missing << 'Distribution.Storage Operation Scheme was not changed to TrackChargeDischargeSchedules' unless cf(dist_text).include?(cf('TrackChargeDischargeSchedules'))
      missing << "Distribution.Storage Converter Object Name was not changed to #{CONVERTER_NAME}" unless cf(dist_text).include?(cf(CONVERTER_NAME))
      missing << "Distribution.Charge Schedule was not changed to #{CHARGE_SCHEDULE_NAME}" unless cf(dist_text).include?(cf(CHARGE_SCHEDULE_NAME))
      missing << "Distribution.Discharge Schedule was not changed to #{DISCHARGE_SCHEDULE_NAME}" unless cf(dist_text).include?(cf(DISCHARGE_SCHEDULE_NAME))
    end

    supervisor = find_object_by_type_and_name(workspace, 'EnergyManagementSystem:Program', PROGRAM_NAME)
    missing << "Missing EMS Program=#{PROGRAM_NAME}" if supervisor.nil?

    pcm = find_object_by_type_and_name(workspace, 'EnergyManagementSystem:ProgramCallingManager', PCM_NAME)
    missing << "Missing EMS ProgramCallingManager=#{PCM_NAME}" if pcm.nil?

    if missing.any?
      runner.registerError("Hybrid island patch validation failed:#{missing.join(' | ')}")
      return false
    end

    runner.registerInfo('Hybrid island patch validation passed.')
    true
  end


  # arguments


  def arguments(_workspace)
    args = OpenStudio::Measure::OSArgumentVector.new

    outage_mode = OpenStudio::Measure::OSArgument.makeBoolArgument('outage_mode', true)
    outage_mode.setDisplayName('Outage/islanding control')
    outage_mode.setDescription('false = disables this measure;true = switches between grid-connected and islanded operation using county and scenario JSON.')
    outage_mode.setDefaultValue(true)
    args << outage_mode

    ssp_choices = OpenStudio::StringVector.new
    ssp_choices << 'ssp126'
    ssp_choices << 'ssp245'
    ssp_choices << 'ssp585'
    ssp_scenario = OpenStudio::Measure::OSArgument.makeChoiceArgument('ssp_scenario', ssp_choices, true)
    ssp_scenario.setDisplayName('SSP scenario')
    ssp_scenario.setDefaultValue('ssp245')
    args << ssp_scenario

    year_choices = OpenStudio::StringVector.new
    year_choices << '2020s'
    year_choices << '2030s'
    year_choices << '2040s'
    year_choices << '2050s'
    scenario_year = OpenStudio::Measure::OSArgument.makeChoiceArgument('scenario_year', year_choices, true)
    scenario_year.setDisplayName('scenarioperiod')
    scenario_year.setDefaultValue('2020s')
    args << scenario_year

    county_code = OpenStudio::Measure::OSArgument.makeStringArgument('county_code', false)
    county_code.setDisplayName('County Code()')
    county_code.setDescription('Optional manual override.When blank, the measure first attempts to resolve the county from EPWFilePath or WeatherStation Name in existing.xml.Supports GISJOIN values such as G2500250,and five-digit FIPS values for example 25025.')
    county_code.setDefaultValue('')
    args << county_code

    json_root_dir = OpenStudio::Measure::OSArgument.makeStringArgument('json_root_dir', false)
    json_root_dir.setDisplayName('outage JSON directory')
    json_root_dir.setDescription('default measure/resources/outage_json.')
    json_root_dir.setDefaultValue('')
    args << json_root_dir

    total_hours = OpenStudio::Measure::OSArgument.makeIntegerArgument('total_hours', true)
    total_hours.setDisplayName('')
    total_hours.setDefaultValue(8760)
    args << total_hours

    battery_initial_soc = OpenStudio::Measure::OSArgument.makeDoubleArgument('battery_initial_soc', true)
    battery_initial_soc.setDisplayName(' SOC')
    battery_initial_soc.setDefaultValue(0.95)
    args << battery_initial_soc

    battery_min_soc = OpenStudio::Measure::OSArgument.makeDoubleArgument('battery_min_soc', true)
    battery_min_soc.setDisplayName(' SOC')
    battery_min_soc.setDefaultValue(0.10)
    args << battery_min_soc

    battery_max_soc = OpenStudio::Measure::OSArgument.makeDoubleArgument('battery_max_soc', true)
    battery_max_soc.setDisplayName(' SOC')
    battery_max_soc.setDefaultValue(0.975)
    args << battery_max_soc

    battery_charge_power_w = OpenStudio::Measure::OSArgument.makeDoubleArgument('battery_charge_power_w', true)
    battery_charge_power_w.setDisplayName('(W)')
    battery_charge_power_w.setDefaultValue(5000.0)
    args << battery_charge_power_w

    battery_discharge_power_w = OpenStudio::Measure::OSArgument.makeDoubleArgument('battery_discharge_power_w', true)
    battery_discharge_power_w.setDisplayName('(W)')
    battery_discharge_power_w.setDefaultValue(5000.0)
    args << battery_discharge_power_w

    critical_misc_w = OpenStudio::Measure::OSArgument.makeDoubleArgument('critical_misc_w', true)
    critical_misc_w.setDisplayName('(W)')
    critical_misc_w.setDescription('for example,,.')
    critical_misc_w.setDefaultValue(46.0)
    args << critical_misc_w

    critical_interior_lighting_w = OpenStudio::Measure::OSArgument.makeDoubleArgument('critical_interior_lighting_w', true)
    critical_interior_lighting_w.setDisplayName('(W)')
    critical_interior_lighting_w.setDefaultValue(44.0)
    args << critical_interior_lighting_w

    critical_hvac_w = OpenStudio::Measure::OSArgument.makeDoubleArgument('critical_hvac_w', true)
    critical_hvac_w.setDisplayName(' HVAC (W)')
    critical_hvac_w.setDefaultValue(1500.0)
    args << critical_hvac_w

    debug_force_outage_start_hour = OpenStudio::Measure::OSArgument.makeIntegerArgument('debug_force_outage_start_hour', true)
    debug_force_outage_start_hour.setDisplayName(':outage')
    debug_force_outage_start_hour.setDescription('. debug_force_outage_duration_hours > 0, island.1  1  1  00:00-01:00.0 .')
    debug_force_outage_start_hour.setDefaultValue(0)
    args << debug_force_outage_start_hour

    debug_force_outage_duration_hours = OpenStudio::Measure::OSArgument.makeIntegerArgument('debug_force_outage_duration_hours', true)
    debug_force_outage_duration_hours.setDisplayName(':outage')
    debug_force_outage_duration_hours.setDescription('.default 0 .')
    debug_force_outage_duration_hours.setDefaultValue(0)
    args << debug_force_outage_duration_hours

    fail_if_no_outage_in_runperiod = OpenStudio::Measure::OSArgument.makeBoolArgument('fail_if_no_outage_in_runperiod', true)
    fail_if_no_outage_in_runperiod.setDisplayName(' RunPeriod outage')
    fail_if_no_outage_in_runperiod.setDefaultValue(false)
    args << fail_if_no_outage_in_runperiod

    missing_choices = OpenStudio::StringVector.new
    missing_choices << 'fail'
    missing_choices << 'not_applicable'
    missing_der_behavior = OpenStudio::Measure::OSArgument.makeChoiceArgument('missing_der_behavior', missing_choices, true)
    missing_der_behavior.setDisplayName('DER missing()')
    missing_der_behavior.setDescription('. measure /PV  NotApplicable Skipping,.')
    missing_der_behavior.setDefaultValue('not_applicable')
    args << missing_der_behavior

    args
  end


  # run


  def run(workspace, runner, user_arguments)
    super(workspace, runner, user_arguments)
    return false unless runner.validateUserArguments(arguments(workspace), user_arguments)

    outage_mode = runner.getBoolArgumentValue('outage_mode', user_arguments)
    ssp_scenario = runner.getStringArgumentValue('ssp_scenario', user_arguments)
    scenario_year = runner.getStringArgumentValue('scenario_year', user_arguments)
    county_code = runner.getStringArgumentValue('county_code', user_arguments)
    json_root_dir = runner.getStringArgumentValue('json_root_dir', user_arguments)
    total_hours = runner.getIntegerArgumentValue('total_hours', user_arguments)
    battery_initial_soc = runner.getDoubleArgumentValue('battery_initial_soc', user_arguments)
    battery_min_soc = runner.getDoubleArgumentValue('battery_min_soc', user_arguments)
    battery_max_soc = runner.getDoubleArgumentValue('battery_max_soc', user_arguments)
    battery_charge_power_w = runner.getDoubleArgumentValue('battery_charge_power_w', user_arguments)
    battery_discharge_power_w = runner.getDoubleArgumentValue('battery_discharge_power_w', user_arguments)
    critical_misc_w = runner.getDoubleArgumentValue('critical_misc_w', user_arguments)
    critical_interior_lighting_w = runner.getDoubleArgumentValue('critical_interior_lighting_w', user_arguments)
    critical_hvac_w = runner.getDoubleArgumentValue('critical_hvac_w', user_arguments)
    debug_force_outage_start_hour = runner.getIntegerArgumentValue('debug_force_outage_start_hour', user_arguments)
    debug_force_outage_duration_hours = runner.getIntegerArgumentValue('debug_force_outage_duration_hours', user_arguments)
    fail_if_no_outage_in_runperiod = runner.getBoolArgumentValue('fail_if_no_outage_in_runperiod', user_arguments)
    _missing_der_behavior = runner.getStringArgumentValue('missing_der_behavior', user_arguments)

    unless outage_mode
      runner.registerAsNotApplicable('outage_mode=off,measure .')
      return true
    end

    if total_hours <= 0
      runner.registerError('total_hours  > 0.')
      return false
    end

    if battery_initial_soc <= 0 || battery_initial_soc > 1
      runner.registerError('battery_initial_soc  (0,1].')
      return false
    end

    if battery_min_soc < 0 || battery_max_soc > 1 || battery_min_soc >= battery_max_soc
      runner.registerError('battery_min_soc / battery_max_soc , 0 <= min < max <= 1.')
      return false
    end

    if battery_charge_power_w <= 0 || battery_discharge_power_w <= 0
      runner.registerError('battery_charge_power_w  battery_discharge_power_w  > 0.')
      return false
    end

    json_root_dir = default_json_root_dir if json_root_dir.to_s.strip.empty?

    runner.registerInitialCondition("HybridIslandingControl :scenario=#{ssp_scenario} #{scenario_year}")

    unless ensure_schedule_type_limits_and_always_on(workspace, runner)
      runner.registerError('Unable to ScheduleTypeLimits / Always On Discrete .')
      return false
    end

    core = nil
    begin
      core = discover_hybrid_core(workspace)
    rescue MissingHybridDerError => e


      runner.registerAsNotApplicable("Skipping Hybrid Islanding :#{e.message}")
      return true
    end

    controls = normalize_optional_controls(discover_optional_controls(workspace))

    runner.registerInfo("found:Distribution=#{core[:distribution_name]}, Battery=#{core[:battery_name]}, Inverter=#{core[:inverter_name]}, PV=#{core[:pv_generator_name] || '<none>'}")
    runner.registerInfo(
      "statistics:" \
      "noncritical_eq=#{controls[:noncritical_eq].size}, " \
      "fridge_eq=#{controls[:fridge_eq].size}, " \
      "interior_lights=#{controls[:interior_lights].size}, " \
      "exterior_lights=#{controls[:exterior_lights].size}, " \
      "bath_sched=#{controls[:bath_fan_schedules].size}, " \
      "range_sched=#{controls[:range_fan_schedules].size}, " \
      "wh_sp=#{controls[:water_heater_setpoints].size}, " \
      "hvac_frac=#{controls[:hvac_fraction_schedules].size}, " \
      "htg_sp=#{controls[:heating_setpoint_schedules].size}, " \
      "clg_sp=#{controls[:cooling_setpoint_schedules].size}"
    )

    resolved_county = resolve_county_code(runner, county_code)
    if resolved_county.nil?
      runner.registerError(
        'Unable to county_code. existing.xml  WeatherStation / EPWFilePath,' \
        ' measure  county_code.'
      )
      return false
    end

    remove_previous_hybrid_patch(workspace, core)

    unless ensure_sql_output_enabled(workspace, runner)
      runner.registerError('Unable to SQL output.')
      return false
    end

    unless ensure_ems_debug_output(workspace, runner)
      runner.registerError('Unable to EMS debug output.')
      return false
    end

    patch_battery_initial_soc(workspace, runner, core, battery_initial_soc)

    grid_schedule = generate_grid_schedule(
      county: resolved_county,
      climate: ssp_scenario,
      period: scenario_year,
      total_hours: total_hours,
      json_dir: json_root_dir,
      runner: runner
    )

    outage_schedule = grid_schedule.map { |x| x == 1 ? 0 : 1 }

    outage_schedule = apply_debug_forced_outage(
      outage_schedule,
      debug_force_outage_start_hour,
      debug_force_outage_duration_hours,
      runner
    )

    sim_start_hour, sim_end_hour = extract_runperiod_hour_range(workspace, runner)
    total_outage_hours = outage_schedule.count(1)
    runperiod_outage_hours = count_ones_in_range(outage_schedule, sim_start_hour, sim_end_hour)

    runner.registerInfo("Outage statistics:=#{total_outage_hours},  RunPeriod=#{runperiod_outage_hours}")

    if total_outage_hours == 0
      runner.registerWarning(' scenario/county generate outage  0.')
    end

    if runperiod_outage_hours == 0
      msg = ' RunPeriod  outage schedule ; island .'
      if fail_if_no_outage_in_runperiod
        runner.registerError(msg)
        return false
      else
        runner.registerWarning(msg)
      end
    end

    island_schedule_idf = build_schedule_compact(
      outage_schedule,
      ISLAND_SCHEDULE_NAME,
      schedule_type_limits: 'Fractional',
      year: 2018,
      total_hours: total_hours
    )

    args_hash = {
      battery_initial_soc: battery_initial_soc,
      battery_min_soc: battery_min_soc,
      battery_max_soc: battery_max_soc,
      battery_charge_power_w: battery_charge_power_w,
      battery_discharge_power_w: battery_discharge_power_w,
      critical_misc_w: critical_misc_w,
      critical_interior_lighting_w: critical_interior_lighting_w,
      critical_hvac_w: critical_hvac_w
    }

    objs = build_hybrid_patch_objects(core, controls, args_hash, island_schedule_idf)
    unless add_idf_objects(workspace, runner, objs)
      runner.registerError(' Hybrid Islanding failed.')
      return false
    end

    unless patch_distribution_object(
      workspace,
      core,
      battery_min_soc: battery_min_soc,
      battery_max_soc: battery_max_soc,
      charge_power_w: battery_charge_power_w,
      discharge_power_w: battery_discharge_power_w,
      runner: runner
    )
      runner.registerError('patch_distribution_object failed.')
      return false
    end

    patch_hvac_availability_objects(workspace, runner)

    unless verify_hybrid_patch(workspace, runner, core)
      return false
    end

    runner.registerValue('hybrid_island_total_outage_hours', total_outage_hours)
    runner.registerValue('hybrid_island_runperiod_outage_hours', runperiod_outage_hours)
    runner.registerValue('hybrid_island_county', resolved_county)

    runner.registerFinalCondition(
      "HybridIslandingControl :county=#{resolved_county}, scenario=#{ssp_scenario} #{scenario_year}; " \
      " outage =#{total_outage_hours}, RunPeriod outage =#{runperiod_outage_hours}; " \
      "Battery=#{core[:battery_name]}; PV=#{core[:pv_generator_name] || 'none'}."
    )

    true
  end
end

HybridIslandingControl.new.registerWithApplication
