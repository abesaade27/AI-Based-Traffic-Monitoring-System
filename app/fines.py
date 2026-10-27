def calculate_fine(detected_classes):
    fines = {
        "no_helmet": 1000,
        "no_seatbelt": 500
    }

    violations = []
    total_fine = 0

    if "no_helmet" in detected_classes:
        violations.append("No Helmet → Rs. 1000")
        total_fine += fines["no_helmet"]

    if "no_seatbelt" in detected_classes:
        violations.append("No Seatbelt → Rs. 500")
        total_fine += fines["no_seatbelt"]

    return violations, total_fine
