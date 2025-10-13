import xml.etree.ElementTree as ET

def parse_xml(file):
    print(f"Processing file: {file}") 
    tree = ET.parse(file)
    root = tree.getroot()
    
    # Extract dataset name and nodes
    dataset = root.find(".//dataset").text
    nodes = []
    for node in root.findall(".//network/nodes/node"):
        node_id = int(node.get("id"))
        cx = float(node.find("cx").text)
        cy = float(node.find("cy").text)
        nodes.append({"id": node_id, "cx": cx, "cy": cy})
    
    # Extract requests
    requests = []
    for request in root.findall(".//requests/request"):
        request_id = int(request.get("id"))
        node_id = int(request.get("node"))
        quantity = float(request.find("quantity").text)
        requests.append({"id": request_id, "node": node_id, "quantity": quantity})
    
    # Extract fleet information
    fleet = []
    for vehicle in root.findall(".//fleet/vehicle_profile"):
        departure_node = int(vehicle.find("departure_node").text)
        arrival_node = int(vehicle.find("arrival_node").text)
        capacity = float(vehicle.find("capacity").text)
        fleet.append({"departure_node": departure_node, "arrival_node": arrival_node, "capacity": capacity})
    
    return {"dataset": dataset, "nodes": nodes, "requests": requests, "fleet": fleet}

