"""Small word lists (gazetteers) that help NLTK's NE chunker on Indian text.

NLTK's maxent chunker was trained on American news (ACE corpus). It often splits
Indian names ("Rahul" PERSON + "Sharma" ORGANIZATION) or labels Indian cities
as people. extractor.py uses these lists to correct its labels.
"""

PLACES = {
    # Indian states and union territories
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal", "Delhi", "New Delhi", "Jammu", "Kashmir", "Ladakh",
    "Puducherry", "Chandigarh", "Lakshadweep",
    # Indian cities
    "Mumbai", "Bengaluru", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad",
    "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal", "Patna", "Surat", "Vadodara",
    "Coimbatore", "Kochi", "Thiruvananthapuram", "Mysuru", "Mysore", "Mangaluru", "Mangalore",
    "Hubballi", "Belagavi", "Visakhapatnam", "Vijayawada", "Madurai", "Noida", "Gurugram",
    "Gurgaon", "Ghaziabad", "Agra", "Varanasi", "Amritsar", "Ludhiana", "Dehradun", "Shimla",
    "Guwahati", "Bhubaneswar", "Ranchi", "Raipur", "Imphal", "Shillong", "Srinagar",
    "Tirupati", "Udupi", "Davanagere", "Tumakuru", "Kozhikode", "Nashik", "Aurangabad",
    "Rajkot", "Jodhpur", "Udaipur", "Allahabad", "Prayagraj", "Meerut", "Jabalpur", "Gwalior",
    # Countries and large world cities
    "India", "Nepal", "Bangladesh", "Sri Lanka", "Pakistan", "China", "Japan", "Singapore",
    "Malaysia", "Australia", "Canada", "Germany", "France", "Italy", "Spain", "Russia",
    "Brazil", "England", "America", "USA", "U.S.", "United States", "UK", "United Kingdom",
    "UAE", "Dubai", "London", "Paris", "Tokyo", "New York", "Sydney", "Toronto", "Berlin",
    "San Francisco", "Seattle", "Boston", "Chicago",
}

KNOWN_ORGS = {
    "Infosys", "Wipro", "TCS", "HCL", "Tata", "Accenture", "Cognizant", "Capgemini", "Deloitte",
    "Mindtree", "Google", "Microsoft", "Amazon", "Apple", "Meta", "IBM", "Intel", "Nvidia",
    "Oracle", "Adobe", "Cisco", "Samsung", "Bosch", "Siemens", "Uber", "Flipkart", "Reliance",
    "Zomato", "Swiggy", "Paytm", "Razorpay", "Zoho", "Freshworks", "Ola", "ISRO", "DRDO", "NASA",
    "BSNL", "SBI", "HDFC", "ICICI", "LIC", "IRCTC", "BCCI", "AICTE", "UGC", "VTU", "IISc",
    "UNESCO", "UNICEF", "WHO", "RBI", "SEBI", "NITI Aayog",
}

# A name containing one of these words is an organization: "Canara Bank", "University of Mysore".
ORG_WORDS = {
    "ltd", "limited", "pvt", "inc", "corp", "corporation", "company", "llp", "bank", "university",
    "college", "institute", "school", "academy", "hospital", "clinic", "technologies",
    "solutions", "services", "systems", "software", "labs", "laboratories", "industries",
    "motors", "airlines", "airways", "railways", "foundation", "trust", "society", "ministry",
    "department", "board", "council", "committee", "commission", "authority", "agency", "club",
    "association", "federation", "enterprises", "pharma", "pharmaceuticals", "bureau",
    "organisation", "organization", "party", "consultancy", "infotech",
}

# Titles mark the next words as a person's name. The title itself is not kept.
TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof", "shri", "sri", "smt", "kumari"}

# Common Indian surnames and first names (a name with one of these is a PERSON).
SURNAMES = {
    "sharma", "verma", "gupta", "singh", "kumar", "patel", "shah", "mehta", "joshi", "reddy",
    "rao", "naidu", "nair", "menon", "pillai", "iyer", "iyengar", "das", "dutta", "bose",
    "banerjee", "chatterjee", "mukherjee", "ghosh", "sen", "yadav", "mishra", "pandey",
    "tiwari", "dubey", "chauhan", "thakur", "agarwal", "jain", "bhat", "hegde", "shetty",
    "kulkarni", "deshpande", "patil", "jadhav", "pawar", "khan", "ahmed", "hussain", "sheikh",
    "fernandes", "dsouza", "kapoor", "malhotra", "chopra", "khanna", "arora", "saxena",
    "srivastava", "bhatt", "krishnan", "subramanian", "gowda", "naik", "kamath", "prasad",
}
FIRST_NAMES = {
    "rahul", "priya", "amit", "anjali", "rohit", "sneha", "vikram", "pooja", "arjun", "kavya",
    "aditya", "divya", "karthik", "ananya", "suresh", "lakshmi", "ravi", "meera", "sanjay",
    "neha", "vijay", "deepa", "arun", "swati", "manoj", "nisha", "rajesh", "sunita", "anil",
    "kiran", "ramesh", "geeta", "mahesh", "rekha", "ajay", "shreya", "nikhil", "aishwarya",
    "varun", "isha", "akash", "riya", "harish", "sowmya", "naveen", "bhavana", "gaurav",
    "tanvi", "abhishek", "nandini", "krithik", "parth", "siddharth", "aarav", "vivek",
}

# Capitalised words that are never entities on their own.
NOT_ENTITIES = {
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "sept", "oct", "nov", "dec", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "rs", "inr", "usd", "gst", "pin", "ist", "am", "pm", "otp", "id",
    "pdf", "ceo", "cto", "cfo", "hr", "ph", "tel", "mob", "dear", "regards", "hi", "hello",
    "thanks", "sir", "madam", "note", "email", "e-mail", "phone",
}
