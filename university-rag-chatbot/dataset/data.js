const PROSPECTUS_TEXT = `=== VIT BHOPAL UNIVERSITY ADMISSIONS PROSPECTUS ===

[Section: Admissions & Eligibility]
VIT Bhopal University admissions for the academic year 2026-27 are open to domestic and international applicants. Eligibility requires a minimum aggregate of 60% in Physics, Chemistry, and Mathematics (PCM) at the 10+2 (high school) level. Admission is strictly based on rank in the VIT Engineering Entrance Examination (VITEEE) or qualifying JEE Main percentile (>85% percentile). Candidates whose 12th board results are awaited can apply and register for counseling. Direct admission without entrance scores is not permitted. The application process is completely online: candidates register on the admissions portal, fill out their profile, upload transcripts, and pay a fee of 1,500 INR. Applications close on August 15, 2026. Official document audits and counseling sessions will begin on August 20, 2026. Selected candidates will receive official email and SMS notifications, and counselor allocation ranks can be viewed on the login dashboard.
For international candidates or NRI students, direct admissions are available based on equivalent qualifying examination scores (e.g. SAT scores or high school transcript averages) without taking VITEEE. Candidates whose date of birth falls on or after July 1, 2004 are eligible to apply. Minor corrections can be requested through the portal before the August 15 deadline. A portion of seats is reserved for state candidates residing in Madhya Pradesh (MP domicile certificates required).

[Section: Tuition Fees & Financial Aid]
The annual tuition fee structure for VIT Bhopal undergraduate and postgraduate courses is distributed across semesters:
- B.Tech in Computer Science & Engineering (CSE) and Data Science & AI: 220,000 INR per year (110,000 INR per semester).
- B.Tech in Electronics & Communication (ECE) and Robotics: 200,000 INR per year (100,000 INR per semester).
- MBA in Business Analytics: 320,000 INR per year (160,000 INR per semester).
- M.Tech in Cybersecurity / VLSI Design: 150,000 INR per year (75,000 INR per semester).
A one-time non-refundable admission registration fee of 10,000 INR is charged at the time of seat acceptance. Semester examination fees are 5,000 INR per semester, which must be paid prior to hall ticket release.
We offer Merit-Based Scholarships: Rank 1-10 in VITEEE get a 100% tuition waiver; Rank 11-100 get a 50% waiver; students scoring >96% in CBSE boards receive a 25% tuition fee concession. Financial aid is available for low-income candidates, allowing payments in up to four interest-free installments per year. All tuition and hostel fees are fully refundable (minus a 1,000 INR processing fee) if withdrawal is requested before classes commence. A special 10% concession on hostel room rent is granted to all female students.

[Section: Hostel & Mess Facilities]
VIT Bhopal University provides premium residential facilities on campus. Accommodation options include single, double, and triple sharing rooms, with options for Air Conditioning (AC) or Non-AC ventilation:
- Single AC Room with attached washroom: 180,000 INR per year.
- Double Sharing AC Room with shared bath: 140,000 INR per year.
- Double Sharing Non-AC Room: 100,000 INR per year.
- Triple Sharing Non-AC Room: 80,000 INR per year.
All hostel packages include a mandatory mess membership offering four healthy meals per day (breakfast, lunch, evening snacks, and dinner), with options for both North Indian and South Indian vegetarian/non-vegetarian cuisine. Facilities include high-speed Wi-Fi, laundry service (up to 40 items per month included in package), gymnasiums, sports courts for basketball, badminton, tennis, and 24/7 CCTV security with biometric locks. Parents can stay in the university guest house for a maximum of 2 days, subject to availability and nominal charges of 1,500 INR per day. Hostels do not allow guests. Any damage to room furniture or fittings will result in fine recovery based on repair costs and disciplinary warnings.

[Section: Academic Branches & Specializations]
The school offers diverse technology and management specializations. The Choice-Based Credit System (CBCS) requires students to complete 160 credits to graduate with a B.Tech degree:
- Computer Science & Engineering (CSE): Focuses on algorithms, software design, cloud architectures, and machine learning. Core courses cover Data Structures, Algorithms, Software Engineering, Database Systems, Cloud Computing, and Machine Learning. Elective options include Internet of Things (IoT), Big Data Analytics, Blockchain, Deep Learning, Mobile App Development, and Cyber Forensics.
- Data Science & Artificial Intelligence (AI): Focuses on neural networks, statistical analysis, natural language processing, and big data systems.
- Electronics & Communication (ECE): Focuses on microprocessors, IoT sensor networks, embedded systems, and telecommunication.
- Business Analytics (MBA): Focuses on corporate finance, operations management, supply chain analytics, and predictive modeling.
Students can also choose minor specializations (18 additional credits) in fields like Fintech, Robotics, or Creative Design alongside their primary major. Academic migration or branch change is permitted after the second semester for top-performing students who have maintained a CGPA above 8.5 with zero backlogs. Students must maintain a CGPA above 5.0 to stay in good academic standing and avoid academic probation. Grades are awarded on a 10-point scale: S (10), A (9), B (8), C (7), D (6), E (5), and F (0/Fail).

[Section: Placement & Internship Records]
VIT Bhopal University boasts an outstanding placement rate of 98.6% for its graduating batch. Placements depend on academic performance, cleared backlogs, and clearing recruiter interview rounds. The average annual CTC package offered is 8.5 Lakhs INR, while the highest package recorded was 44 Lakhs INR by Microsoft. Top recruiting companies visiting the campus include Google, Amazon, Microsoft, NVIDIA, Cognizant, and Deloitte. A mandatory six-month internship in the 7th or 8th semester is a graduation requirement for all engineering streams. The university's dedicated Corporate Relations Cell assists students in securing internships with stipends ranging from 20,000 INR to 80,000 INR per month. The placement cell also runs mandatory aptitude training, coding bootcamps, resume reviews, and mock technical/HR interviews starting from the 5th and 6th semesters. Most top recruiters require zero active backlogs, though some companies permit up to one active backlog.`;

const DATASET_DATA = [
  {
    "id": "qa_1",
    "question": "What is the minimum percentage required for B.Tech admission?",
    "answer": "VIT Bhopal University eligibility requires a minimum aggregate of 60% in Physics, Chemistry, and Mathematics (PCM) at the 10+2 (high school) level.",
    "category": "Admissions"
  },
  {
    "id": "qa_2",
    "question": "Which entrance exams are accepted for B.Tech admission?",
    "answer": "Admission is strictly based on the candidate's rank in the VIT Engineering Entrance Examination (VITEEE) or a qualifying JEE Main percentile (>85% percentile).",
    "category": "Admissions"
  },
  {
    "id": "qa_3",
    "question": "What is the application fee and deadline?",
    "answer": "The online application fee is 1,500 INR. Applications for the 2026-27 academic year close on August 15, 2026.",
    "category": "Admissions"
  },
  {
    "id": "qa_4",
    "question": "When does counseling and verification begin?",
    "answer": "Official document audits and counseling sessions for engineering/science branches will begin on August 20, 2026.",
    "category": "Admissions"
  },
  {
    "id": "qa_5",
    "question": "Is there direct admission without entrance exams?",
    "answer": "No, direct admission is not available for B.Tech programs. All candidates must qualify through VITEEE rank or score above the 85th percentile in JEE Main.",
    "category": "Admissions"
  },
  {
    "id": "qa_6",
    "question": "What is the criteria for NRI/Foreign student admission?",
    "answer": "NRI and foreign students are eligible for direct admission based on their performance in equivalent qualifying examinations (e.g., SAT scores or high school transcript averages) without taking VITEEE.",
    "category": "Admissions"
  },
  {
    "id": "qa_7",
    "question": "Can I edit my application form after submission?",
    "answer": "Yes, minor corrections (like spelling errors or updating marks) can be requested through the admissions support portal before the final deadline of August 15, 2026.",
    "category": "Admissions"
  },
  {
    "id": "qa_8",
    "question": "Is there an age limit for B.Tech admission?",
    "answer": "Yes, candidates whose date of birth falls on or after July 1, 2004, are eligible to apply for B.Tech admissions for the academic year 2026-27.",
    "category": "Admissions"
  },
  {
    "id": "qa_9",
    "question": "Is there any reservation for state candidates (Madhya Pradesh domicile)?",
    "answer": "Yes, VIT Bhopal University reserves a portion of seats for candidates residing in Madhya Pradesh. MP domicile certificates must be submitted during document verification.",
    "category": "Admissions"
  },
  {
    "id": "qa_10",
    "question": "Can I apply for B.Tech admission if my 12th board results are awaited?",
    "answer": "Yes, candidates awaiting board results can apply and register for counseling. Final admission is subject to meeting the 60% PCM eligibility once results are announced.",
    "category": "Admissions"
  },
  {
    "id": "qa_11",
    "question": "How do I check my selection status for counseling?",
    "answer": "Selected candidates will receive official email and SMS notifications. Selection ranks and counselor allocation links can be viewed in the student login dashboard.",
    "category": "Admissions"
  },
  {
    "id": "qa_12",
    "question": "What is the fee structure for B.Tech in Computer Science & Engineering?",
    "answer": "The annual tuition fee for B.Tech in CSE (and B.Tech in Data Science & AI) is 220,000 INR per year, which is paid as 110,000 INR per semester.",
    "category": "Fees"
  },
  {
    "id": "qa_13",
    "question": "What is the cost of Electronics & Communication (ECE) branch?",
    "answer": "The annual tuition fee for ECE (and B.Tech in Robotics) is 200,000 INR per year (or 100,000 INR per semester).",
    "category": "Fees"
  },
  {
    "id": "qa_14",
    "question": "Are there any scholarships available for top rankers?",
    "answer": "Yes, Rank 1-10 in VITEEE get a 100% tuition fee waiver; Rank 11-100 get a 50% waiver; and students scoring >96% in CBSE boards receive a 25% tuition concession.",
    "category": "Fees"
  },
  {
    "id": "qa_15",
    "question": "Can I pay tuition fees in installments?",
    "answer": "Yes, financial aid is available for low-income candidates, which permits paying the annual fees in up to four interest-free installments.",
    "category": "Fees"
  },
  {
    "id": "qa_16",
    "question": "What is the tuition fee for MBA in Business Analytics?",
    "answer": "The annual tuition fee for the MBA in Business Analytics is 320,000 INR, which is paid as 160,000 INR per semester.",
    "category": "Fees"
  },
  {
    "id": "qa_17",
    "question": "Is the hostel fee refundable if I withdraw my admission?",
    "answer": "Yes, hostel and tuition fees are fully refundable (minus a small processing charge of 1,000 INR) if the withdrawal is requested before the start of classes.",
    "category": "Fees"
  },
  {
    "id": "qa_18",
    "question": "What is the tuition fee for M.Tech programs?",
    "answer": "The annual tuition fee for M.Tech in Cybersecurity or VLSI Design is 150,000 INR per year (or 75,000 INR per semester).",
    "category": "Fees"
  },
  {
    "id": "qa_19",
    "question": "Is there any discount for female students?",
    "answer": "Yes, under the university's diversity initiative, female students receive a special 10% concession on hostel room rent across all categories.",
    "category": "Fees"
  },
  {
    "id": "qa_20",
    "question": "Are there any administrative or registration charges in the first year?",
    "answer": "Yes, a one-time non-refundable admission registration fee of 10,000 INR is charged at the time of seat acceptance.",
    "category": "Fees"
  },
  {
    "id": "qa_21",
    "question": "What is the fee for the examination per semester?",
    "answer": "Semester examination fees are 5,000 INR per semester, which must be paid online prior to the release of hall tickets.",
    "category": "Fees"
  },
  {
    "id": "qa_22",
    "question": "What is the cost of a single sharing AC room in the hostel?",
    "answer": "A Single AC Room with an attached washroom costs 180,000 INR per year. It includes high-speed Wi-Fi, laundry, mess dining, and utilities.",
    "category": "Hostel"
  },
  {
    "id": "qa_23",
    "question": "What are the prices for shared rooms in the hostel?",
    "answer": "A Double Sharing AC room costs 140,000 INR per year; a Double Sharing Non-AC room costs 100,000 INR per year; and a Triple Sharing Non-AC room costs 80,000 INR per year.",
    "category": "Hostel"
  },
  {
    "id": "qa_24",
    "question": "Is mess food included in the hostel fee, and what is the menu?",
    "answer": "Yes, a mandatory 4-meal daily mess subscription is included, serving North/South Indian vegetarian and non-vegetarian options (breakfast, lunch, snacks, dinner).",
    "category": "Hostel"
  },
  {
    "id": "qa_25",
    "question": "What are the sports and gym facilities in hostels?",
    "answer": "Hostels feature fully-equipped modern indoor gymnasiums, sports courts for basketball, badminton, tennis, and 24/7 access to running tracks.",
    "category": "Hostel"
  },
  {
    "id": "qa_26",
    "question": "What is the cost of a double sharing Non-AC room?",
    "answer": "A Double Sharing Non-AC room costs 100,000 INR per year. This includes laundry service, high-speed Wi-Fi, and 4 meals daily in the mess.",
    "category": "Hostel"
  },
  {
    "id": "qa_27",
    "question": "What is the cost of a triple sharing Non-AC room?",
    "answer": "A Triple Sharing Non-AC room is the most economical option, costing 80,000 INR per year, which includes all mess meals and campus services.",
    "category": "Hostel"
  },
  {
    "id": "qa_28",
    "question": "Is laundry service included in the hostel fee?",
    "answer": "Yes, professional steam laundry and washing cycles (up to 40 items of clothing per month) are included in the hostel fee packages.",
    "category": "Hostel"
  },
  {
    "id": "qa_29",
    "question": "What is the cost of a double sharing AC room in the hostel?",
    "answer": "A Double Sharing AC room costs 140,000 INR per year. It includes high-speed Wi-Fi, laundry service, and 4 daily mess meals.",
    "category": "Hostel"
  },
  {
    "id": "qa_30",
    "question": "Are guests or parents allowed to stay in the university hostel?",
    "answer": "Parents can stay in the university guest house for a maximum of 2 days, subject to availability and nominal charges (1,500 INR/day). Hostels do not allow guests.",
    "category": "Hostel"
  },
  {
    "id": "qa_31",
    "question": "What is the penalty for damaging hostel property?",
    "answer": "Any damage to room furniture, electrical fittings, or sanitaryware will result in fine recovery based on repair costs and disciplinary warnings.",
    "category": "Hostel"
  },
  {
    "id": "qa_32",
    "question": "How many credits are required to graduate in B.Tech?",
    "answer": "Under the Choice-Based Credit System (CBCS), students must successfully complete 160 credits to graduate with a B.Tech degree.",
    "category": "Academics"
  },
  {
    "id": "qa_33",
    "question": "Can I take minor courses alongside my major?",
    "answer": "Yes, students can take minor specializations (requiring 18 additional credits) in fields like Fintech, Robotics, or Creative Design.",
    "category": "Academics"
  },
  {
    "id": "qa_34",
    "question": "What is the Choice-Based Credit System (CBCS)?",
    "answer": "CBCS allows students to choose their own courses, theoretical subjects, practical labs, and electives from a diverse pool to complete their 160-credit graduation requirement.",
    "category": "Academics"
  },
  {
    "id": "qa_35",
    "question": "What are the core subjects in Computer Science (CSE)?",
    "answer": "Core subjects include Data Structures, Design & Analysis of Algorithms, Software Engineering, Database Systems, Cloud Computing, and Machine Learning.",
    "category": "Academics"
  },
  {
    "id": "qa_36",
    "question": "How many credits does a minor specialization require?",
    "answer": "Minor specializations (in fields like Fintech, Robotics, or Creative Design) require 18 additional credits alongside your primary major.",
    "category": "Academics"
  },
  {
    "id": "qa_37",
    "question": "Can I request a branch change after the first year?",
    "answer": "Yes, branch migration is permitted at the end of the second semester for top-performing students who have maintained a CGPA above 8.5 with no backlogs.",
    "category": "Academics"
  },
  {
    "id": "qa_38",
    "question": "What electives are offered in B.Tech Computer Science?",
    "answer": "Electives include Internet of Things (IoT), Big Data Analytics, Blockchain Technologies, Deep Learning, Mobile App Development, and Cyber Forensics.",
    "category": "Academics"
  },
  {
    "id": "qa_39",
    "question": "What is the minimum CGPA required to avoid academic probation?",
    "answer": "Students must maintain a Cumulative Grade Point Average (CGPA) above 5.0 to stay in good academic standing and avoid probation.",
    "category": "Academics"
  },
  {
    "id": "qa_40",
    "question": "How are grades calculated in the Choice-Based Credit System?",
    "answer": "Grades are awarded on a 10-point scale: S (10 points), A (9 points), B (8 points), C (7 points), D (6 points), E (5 points), and F (0 points/Fail).",
    "category": "Academics"
  },
  {
    "id": "qa_41",
    "question": "What is the placement percentage and average salary package?",
    "answer": "VIT Bhopal University has a 98.6% placement rate, with an average CTC of 8.5 Lakhs INR per year. The highest package reached 44 Lakhs INR offered by Microsoft.",
    "category": "Placements"
  },
  {
    "id": "qa_42",
    "question": "Is an internship mandatory for engineering students?",
    "answer": "Yes, a compulsory six-month internship in the 7th or 8th semester is required. The stipend ranges from 20,000 INR to 80,000 INR per month.",
    "category": "Placements"
  },
  {
    "id": "qa_43",
    "question": "Who are the top recruiters at VIT Bhopal?",
    "answer": "Top recruiting companies visiting the campus include Microsoft, Google, Amazon, NVIDIA, Cognizant, and Deloitte. Microsoft offered the highest package of 44 Lakhs INR.",
    "category": "Placements"
  },
  {
    "id": "qa_44",
    "question": "What is the highest salary package offered, and by which company?",
    "answer": "The highest salary package recorded is 44 Lakhs INR per year, offered by Microsoft to a B.Tech Computer Science student.",
    "category": "Placements"
  },
  {
    "id": "qa_45",
    "question": "What is the average stipend for six-month internships?",
    "answer": "The average monthly stipend for engineering students ranges from 20,000 INR to 80,000 INR per month, depending on the recruiter.",
    "category": "Placements"
  },
  {
    "id": "qa_46",
    "question": "Is there a placement cell for student training?",
    "answer": "Yes, the Corporate Relations Cell runs mandatory aptitude training, coding bootcamps, and mock interview drills starting from the 5th semester.",
    "category": "Placements"
  },
  {
    "id": "qa_47",
    "question": "Does the university guarantee placements for all students?",
    "answer": "While the placement rate is exceptionally high at 98.6%, placements depend on academic performance, cleared backlogs, and clearing recruiter interview rounds.",
    "category": "Placements"
  },
  {
    "id": "qa_48",
    "question": "Can I apply for placement drives if I have active backlogs?",
    "answer": "Most top recruiters require zero active backlogs. However, some companies permit candidates with up to one active backlog to participate in selection rounds.",
    "category": "Placements"
  },
  {
    "id": "qa_49",
    "question": "Are there mock interviews conducted before actual placement drives?",
    "answer": "Yes, the Corporate Relations Cell runs compulsory mock technical interviews, HR interviews, and resume reviews starting from the 6th semester.",
    "category": "Placements"
  },
  {
    "id": "qa_50",
    "question": "Who guides students for off-campus placement opportunities?",
    "answer": "The Corporate Relations Cell provides off-campus circulars, verification assistance, and mock interview support for students applying independently.",
    "category": "Placements"
  }
];
