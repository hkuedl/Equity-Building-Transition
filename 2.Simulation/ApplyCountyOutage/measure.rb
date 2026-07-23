require 'openstudio'
require 'json'
require 'rexml/document'

class ApplyCountyOutage < OpenStudio::Measure::ModelMeasure
  def name
    return "Apply County Outage"
  end

  def arguments(model)
    args = OpenStudio::Measure::OSArgumentVector.new


    outage_mode = OpenStudio::Measure::OSArgument.makeChoiceArgument("outage_mode", ["on", "off"], true)
    outage_mode.setDisplayName("Outage Mode")
    outage_mode.setDefaultValue("off")
    args << outage_mode


    ssp_scenario = OpenStudio::Measure::OSArgument.makeChoiceArgument("ssp_scenario", ["ssp126", "ssp245", "ssp585"], true)
    ssp_scenario.setDisplayName("SSP Scenario")
    ssp_scenario.setDefaultValue("ssp245")
    args << ssp_scenario


    scenario_year = OpenStudio::Measure::OSArgument.makeChoiceArgument("scenario_year", ["2020s", "2030s", "2040s", "2050s"], true)
    scenario_year.setDisplayName("Scenario Year")
    scenario_year.setDefaultValue("2020s")
    args << scenario_year

    return args
  end


  def get_time_from_idx(idx)

    doy = (idx / 24).floor + 1
    hour = idx % 24


    if doy > 365
      return 12, 31, 23
    end

    os_date = OpenStudio::Date.fromDayOfYear(doy)
    return os_date.monthOfYear.value, os_date.dayOfMonth, hour
  end

  def run(model, runner, user_arguments)
    super(model, runner, user_arguments)

    if !runner.validateUserArguments(arguments(model), user_arguments)
      return false
    end

    outage_mode = runner.getStringArgumentValue("outage_mode", user_arguments)
    ssp_scenario = runner.getStringArgumentValue("ssp_scenario", user_arguments)
    scenario_year = runner.getStringArgumentValue("scenario_year", user_arguments)

    if outage_mode == "off"
      runner.registerInfo("Outage mode is set to off. Skipping.")
      return true
    end


    parent_dir = File.expand_path("..")
    target_xml_path = nil
    ["upgraded.xml", "home.xml", "in.xml", "existing.xml"].each do |f|
      p = File.join(parent_dir, f)
      if File.exist?(p) then target_xml_path = p; break end
    end

    if target_xml_path.nil?
      runner.registerError(" Could not find HPXML file to patch.")
      return false
    end

    xml_content = File.read(target_xml_path)
    current_id = xml_content =~ /(G\d{7})/ ? $1 : nil
    if current_id.nil?
      runner.registerError(" No Gxxxxxxx ID found in XML.")
      return false
    end


    json_filename = "#{ssp_scenario}_#{scenario_year}_Scenario_EPW_Scenario_EPW.json"
    resource_path = File.join(File.dirname(__FILE__), "resources", json_filename)

    if !File.exist?(resource_path)
      runner.registerError(" JSON missing at: #{resource_path}")
      return false
    end

    outage_periods = []
    begin

      file_content = File.read(resource_path)
      outage_data = JSON.parse(file_content)


      if !outage_data.key?(current_id)
        runner.registerError(" CRITICAL: ID #{current_id} is missing from #{json_filename}. This simulation is aborted to ensure data integrity.")
        return false
      end

      outage_periods = outage_data[current_id]


      if outage_periods.empty?
        runner.registerInfo(" Building #{current_id} confirmed to have NO outages in this scenario. Proceeding normally.")
        return true
      end


      runner.registerInfo(" Building #{current_id} has #{outage_periods.length} scheduled outage periods.")

    rescue => e
      runner.registerError(" JSON/Data Error: #{e.message}")
      return false
    end


    doc = REXML::Document.new(xml_content)
    software_info = doc.elements["/HPXML/SoftwareInfo"]
    if software_info.nil?
      runner.registerError(" Invalid HPXML: Missing SoftwareInfo element.")
      return false
    end

    extension = software_info.elements["extension"] || software_info.add_element("extension")
    extension.delete_element("UnavailablePeriods") if extension.elements["UnavailablePeriods"]
    up_tag = extension.add_element("UnavailablePeriods")

    outage_periods.each do |start_idx, end_idx|
      sm, sd, sh = get_time_from_idx(start_idx)

      em, ed, eh = get_time_from_idx(end_idx + 1)

      u = up_tag.add_element("UnavailablePeriod")
      u.add_element("ColumnName").text = "Power Outage"
      u.add_element("BeginMonth").text = sm.to_s
      u.add_element("BeginDayOfMonth").text = sd.to_s
      u.add_element("BeginHourOfDay").text = sh.to_s
      u.add_element("EndMonth").text = em.to_s
      u.add_element("EndDayOfMonth").text = ed.to_s
      u.add_element("EndHourOfDay").text = eh.to_s
    end


    File.open(target_xml_path, "w") { |f| doc.write(f) }
    runner.registerInfo(" Applied #{outage_periods.length} outage periods from #{json_filename} to #{current_id}.")
    return true
  end
end
