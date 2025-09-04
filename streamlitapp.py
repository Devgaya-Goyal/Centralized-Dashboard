from botocore.exceptions import ClientError
from openai import OpenAI
import streamlit as st
import psutil
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFilter
import smtplib
import cv2
import os
import pywhatkit
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage  
import tempfile
import numpy as np
import random
import boto3
import subprocess
import paramiko
import time
import webbrowser

def detect_face(image):
    """
    Detects a face in the provided image using OpenCV's Haar Cascade classifier.
    Returns the coordinates (x1, y1, x2, y2) of the first detected face.
    """
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return None
            
        # Return the first face coordinates in (x1, y1, x2, y2) format
        x, y, w, h = faces[0]
        return (x, y, x + w, y + h)
        
    except Exception as e:
        print(f"Face detection error: {e}")
        return None

def create_feathered_mask(size, coords, feather_radius):
    """
    Creates a feathered circular mask for a given coordinate.
    """
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(coords, fill=255)
    return mask.filter(ImageFilter.GaussianBlur(feather_radius))

def swap_faces(img1, face_coords1, img2, face_coords2):
    """
    Swaps faces between two images using a feathered mask for a smooth blend.
    """
    # Extract face regions
    face_region1 = img1.crop(face_coords1)
    face_region2 = img2.crop(face_coords2)

    # Get face dimensions
    face_width1, face_height1 = face_coords1[2] - face_coords1[0], face_coords1[3] - face_coords1[1]
    face_width2, face_height2 = face_coords2[2] - face_coords2[0], face_coords2[3] - face_coords2[1]

    # Resize faces to fit the target regions
    face_region2_resized = face_region2.resize((face_width1, face_height1))
    face_region1_resized = face_region1.resize((face_width2, face_height2))
    
    # Create feathered masks for pasting
    feather_radius = int(min(face_width1, face_height1) * 0.2)
    mask1 = create_feathered_mask(img1.size, face_coords1, feather_radius)
    
    feather_radius = int(min(face_width2, face_height2) * 0.2)
    mask2 = create_feathered_mask(img2.size, face_coords2, feather_radius)

    # Perform the face swaps using the feathered masks
    swapped_img1 = img1.copy()
    swapped_img1.paste(face_region2_resized, (face_coords1[0], face_coords1[1]), mask1.crop(face_coords1))

    swapped_img2 = img2.copy()
    swapped_img2.paste(face_region1_resized, (face_coords2[0], face_coords2[1]), mask2.crop(face_coords2))

    return swapped_img1, swapped_img2

def create_digital_image():
    try:
        img_width = 1200
        img_height = 800
        img = Image.new('RGB', (img_width, img_height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        tile_size = 50

        for x in range(0, img_width, tile_size):
            for y in range(0, img_height, tile_size):
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                fill_color = (r, g, b)
                draw.rectangle([x, y, x + tile_size, y + tile_size], fill=fill_color)

        output_filename = "mosaic_image.png"
        img.save(output_filename)
        print(f"Image saved as '{output_filename}' successfully!")
        return img

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def execute_ssh_command(ssh_client, command):
    try:
        stdin, stdout, stderr = ssh_client.exec_command(command)
        output = stdout.read().decode()
        error = stderr.read().decode()
        return output, error
    except Exception as e:
        return "", str(e)

st.set_page_config(page_title="Centralized Dashboard For Numerous Technologies", layout="wide")

st.sidebar.title("Navigation")
section = st.sidebar.radio("Select Section", ["Home", "Web-App", "Linux", "Docker", "AWS", "Projects"])

if section == "Home":
    st.title("Centralized Dashboard For Numerous Technologies")
    st.markdown("---")
    
    st.header("Project Overview")
    st.write("""
    This comprehensive dashboard serves as a unified platform for managing and executing various 
    technology-related tasks across multiple domains. The dashboard integrates automation tools, 
    cloud services, containerization technologies, and development workflows into a single interface.
    
    Built to streamline operations across different technology stacks, this platform provides 
    seamless access to web applications, system administration, cloud infrastructure management, 
    and automated deployment pipelines.
    """)
    
    st.subheader("Available Sections")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Web-App Section**")
        st.write("- System monitoring and resource tracking")
        st.write("- Communication tools (WhatsApp, SMS, Email)")
        st.write("- Image processing and face recognition")
        st.write("- Web scraping and search automation")
        st.write("- Social media integration")
        
        st.markdown("**Linux Section**")
        st.write("- Remote SSH command execution")
        st.write("- System administration tasks")
        st.write("- File and directory management")
        st.write("- Network configuration")
        st.write("- Process and service management")
        
        st.markdown("**Docker Section**")
        st.write("- Container lifecycle management")
        st.write("- Image building and deployment")
        st.write("- Apache server containerization")
        st.write("- Multi-container orchestration")
    
    with col2:
        st.markdown("**AWS Section**")
        st.write("- EC2 instance management")
        st.write("- CloudWatch logs monitoring")
        st.write("- S3 bucket operations and automation")
        st.write("- Lambda function integration")
        st.write("- MongoDB service connectivity")
        st.write("- Audio transcription workflows")
        
        st.markdown("**Projects Section**")
        st.write("- ChatGPT automation agents")
        st.write("- Cloud infrastructure automation")
        st.write("- Containerized web services")
        st.write("- CI/CD pipeline implementation")
        st.write("- End-to-end deployment solutions")
    
    st.markdown("---")
    st.info("Navigate through the sections using the sidebar to access specific functionalities and tools.")

elif section == "Web-App":
    st.title("Web Application Tasks")
    
    task = st.selectbox("Choose a Task", [
        "System Monitoring",
        "Communication Tools", 
        "Image Processing",
        "Web Automation",
        "Social Media Tools"
    ])
    
    if task == "System Monitoring":
        st.subheader("System Resource Monitoring")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Check RAM Usage"):
                ram = psutil.virtual_memory()
                st.metric("Total RAM", f"{ram.total / (1024**3):.2f} GB")
                st.metric("Available RAM", f"{ram.available / (1024**3):.2f} GB")
                st.metric("RAM Usage", f"{ram.percent}%")
        
        with col2:
            if st.button("Check CPU Usage"):
                cpu_percent = psutil.cpu_percent(interval=1)
                st.metric("CPU Usage", f"{cpu_percent}%")
                cpu_count = psutil.cpu_count()
                st.metric("CPU Cores", cpu_count)
        
        with col3:
            if st.button("Check Disk Usage"):
                try:
                    if os.name == 'nt':
                        disk = psutil.disk_usage('C:\\')
                    else:
                        disk = psutil.disk_usage('/')
                    st.metric("Total Disk", f"{disk.total / (1024**3):.2f} GB")
                    st.metric("Free Disk", f"{disk.free / (1024**3):.2f} GB")
                    st.metric("Disk Usage", f"{(disk.used/disk.total)*100:.1f}%")
                except Exception as e:
                    st.error(f"Error checking disk usage: {e}")

    elif task == "Communication Tools":
        comm_type = st.radio("Select Communication Type", ["WhatsApp", "SMS", "Email", "Phone Call"])
        
        if comm_type == "WhatsApp":
            to_number = st.text_input("To (include country code, e.g., +91...):")
            msg = st.text_area("Message:")
            if st.button("Send WhatsApp"):
                try:
                    pywhatkit.sendwhatmsg_instantly(to_number, msg)
                    st.success("WhatsApp message sent successfully!")
                except Exception as e:
                    st.error(f"Error sending WhatsApp message: {e}")
        
        elif comm_type == "SMS":
            st.info("SMS functionality requires Twilio configuration")
            to = st.text_input("To (Phone Number):")
            body = st.text_area("SMS Message:")
            account_sid = st.text_input("Twilio Account SID:", type="password")
            auth_token = st.text_input("Twilio Auth Token:", type="password")
            from_number = st.text_input("From Number (Twilio):")
            
            if st.button("Send SMS"):
                if account_sid and auth_token and from_number:
                    st.info("SMS integration configured - would send message")
                else:
                    st.warning("Please provide Twilio credentials")
        
        elif comm_type == "Email":
            to_email = st.text_input("To Email:")
            subject = st.text_input("Subject:")
            body = st.text_area("Email Body:")
            sender_email = st.text_input("Your Email:")
            sender_password = st.text_input("Your Email Password:", type="password")
            
            if st.button("Send Email"):
                if sender_email and sender_password:
                    try:
                        msg = MIMEMultipart()
                        msg['From'] = sender_email
                        msg['To'] = to_email
                        msg['Subject'] = subject
                        msg.attach(MIMEText(body, 'plain'))
                        
                        server = smtplib.SMTP('smtp.gmail.com', 587)
                        server.starttls()
                        server.login(sender_email, sender_password)
                        server.send_message(msg)
                        server.quit()
                        st.success("Email sent successfully!")
                    except Exception as e:
                        st.error(f"Error sending email: {e}")
                else:
                    st.warning("Please provide email credentials")
        
        elif comm_type == "Phone Call":
            st.info("Phone call functionality requires Twilio configuration")
            to = st.text_input("To (Phone Number):")
            if st.button("Make Call"):
                st.info("Phone call integration would be initiated here")

    elif task == "Image Processing":
        img_task = st.radio("Select Image Task", ["Create Digital Image", "Face Swap"])
        
        if img_task == "Create Digital Image":
            if st.button("Generate Mosaic Image"):
                mosaic_img = create_digital_image()
                if mosaic_img:
                    st.image(mosaic_img, caption='Generated Mosaic Image')
                    st.success("Mosaic image generated successfully!")
                else:
                    st.error("Failed to generate the mosaic image.")
        
        elif img_task == "Face Swap":
            uploaded_file1 = st.file_uploader("Choose the first image...", type=["jpg", "png", "jpeg"], key="img1")
            uploaded_file2 = st.file_uploader("Choose the second image...", type=["jpg", "png", "jpeg"], key="img2")

            if uploaded_file1 and uploaded_file2:
                col1, col2 = st.columns(2)
                
                img1 = Image.open(uploaded_file1).convert("RGB")
                img2 = Image.open(uploaded_file2).convert("RGB")

                with col1:
                    st.image(img1, caption='Image 1', use_column_width=True)
                with col2:
                    st.image(img2, caption='Image 2', use_column_width=True)

                if st.button("Swap Faces"):
                    with st.spinner('Detecting faces and swapping...'):
                        face_coords1 = detect_face(img1)
                        face_coords2 = detect_face(img2)
                        
                        if face_coords1 and face_coords2:
                            swapped_img1, swapped_img2 = swap_faces(img1, face_coords1, img2, face_coords2)

                            st.subheader("Swapped Faces")
                            col3, col4 = st.columns(2)
                            
                            with col3:
                                st.image(swapped_img1, caption='Swapped Image 1', use_column_width=True)
                            with col4:
                                st.image(swapped_img2, caption='Swapped Image 2', use_column_width=True)
                        else:
                            st.error("Could not detect a face in one or both of the images. Please try with different images.")
    
    elif task == "Web Automation":
        web_task = st.radio("Select Web Task", ["Google Search", "Website Download"])
        
        if web_task == "Google Search":
            query = st.text_input("Search Query:")
            url = f"https://www.google.com/search?q={query}"
            if st.button("Click for results",on_click=lambda: webbrowser.open(url)):
                st.success("Results opened in browser")
            else:
                st.warning("Please enter a search query")
    
        elif web_task == "Website Download":
            url = st.text_input("Website URL:")
            if st.button("Download"):
                if url:
                    try:
                        response = requests.get(url)
                        response.raise_for_status()
                        
                        filename = "website_data.html"
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write(response.text)
                        
                        st.success(f"Website data downloaded as {filename}")
                        st.download_button(
                            label="Download HTML File",
                            data=response.text,
                            file_name=filename,
                            mime="text/html"
                        )
                    except Exception as e:
                        st.error(f"Error downloading website: {e}")
                else:
                    st.warning("Please enter a valid URL")
    
    elif task == "Social Media Tools":
        st.subheader("Social Media Integration")
        
        platform = st.radio("Select Platform", ["Instagram Simulation", "Twitter Simulation"])
        
        if platform == "Instagram Simulation":
            caption = st.text_input("Caption:")
            uploaded_file = st.file_uploader("Choose image to simulate post:", type=["jpg", "png"])
            
            if st.button("Simulate Instagram Post"):
                if caption and uploaded_file:
                    st.success(f"Simulated Instagram post with caption: {caption}")
                    st.image(uploaded_file, caption="Posted Image")
                else:
                    st.warning("Please provide both caption and image")
        
        elif platform == "Twitter Simulation":
            tweet_text = st.text_area("Tweet Text:", max_chars=280)
            st.write(f"Characters: {len(tweet_text)}/280")
            
            if st.button("Simulate Tweet"):
                if tweet_text:
                    st.success(f"Simulated tweet: {tweet_text}")
                else:
                    st.warning("Please enter tweet text")

elif section == "Linux":
    st.title("Linux Command Manager")
    
    st.subheader("SSH Connection")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hostname = st.text_input("Server IP", placeholder="e.g. 192.168.1.10")
    with col2:
        username = st.text_input("Username", value="root")
    with col3:
        password = st.text_input("Password", type="password")

    if "ssh_client" not in st.session_state:
        st.session_state.ssh_client = None

    if st.button("Connect to Server"):
        if hostname and username and password:
            try:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(hostname, username=username, password=password, timeout=10)
                st.session_state.ssh_client = ssh
                st.success(f"Connected to {hostname} as {username}")
            except Exception as e:
                st.error(f"SSH connection failed: {e}")
                st.session_state.ssh_client = None
        else:
            st.warning("Please fill in all SSH connection details")

    st.markdown("---")
    
    linux_categories = [
        "System Information",
        "File Operations", 
        "Network Operations",
        "Process Management",
        "Service Management",
        "Custom Command"
    ]
    
    category = st.selectbox("Select Command Category", linux_categories)
    
    if category == "System Information":
        sys_commands = {
            "Get Date": "date",
            "Get Calendar": "cal",
            "Check Disk Usage": "df -h",
            "Check Memory Usage": "free -h",
            "Check CPU Info": "lscpu",
            "Check System Uptime": "uptime",
            "Check Hostname": "hostname",
            "Check Kernel Version": "uname -r",
            "Check Mounted Filesystems": "mount"
        }
        
        selected_cmd = st.selectbox("Select System Command", list(sys_commands.keys()))
        
        if st.button("Execute System Command"):
            if st.session_state.ssh_client:
                command = sys_commands[selected_cmd]
                output, error = execute_ssh_command(st.session_state.ssh_client, command)
                
                if output:
                    st.success("Command executed successfully:")
                    st.code(output, language="bash")
                if error:
                    st.error(f"Error: {error}")
            else:
                st.warning("Please establish SSH connection first")
    
    elif category == "Network Operations":
        net_commands = {
            "Check Network Configuration": "ifconfig",
            "Check Network Connectivity": "ping -c 4 google.com",
            "Check Open Ports": "netstat -tuln",
            "Check Network Interfaces": "ip addr show",
            "Check Routing Table": "route -n",
            "Check DNS Configuration": "cat /etc/resolv.conf"
        }
        
        selected_cmd = st.selectbox("Select Network Command", list(net_commands.keys()))
        
        if st.button("Execute Network Command"):
            if st.session_state.ssh_client:
                command = net_commands[selected_cmd]
                output, error = execute_ssh_command(st.session_state.ssh_client, command)
                
                if output:
                    st.success("Command executed successfully:")
                    st.code(output, language="bash")
                if error:
                    st.error(f"Error: {error}")
            else:
                st.warning("Please establish SSH connection first")
    
    elif category == "Process Management":
        process_commands = {
            "List Running Processes": "ps aux",
            "Check Top Processes": "top -b -n 1",
            "Check Memory Usage by Process": "ps aux --sort=-%mem | head -10",
            "Check CPU Usage by Process": "ps aux --sort=-%cpu | head -10",
            "Check System Load": "uptime",
            "Check Process Tree": "pstree"
        }
        
        selected_cmd = st.selectbox("Select Process Command", list(process_commands.keys()))
        
        if st.button("Execute Process Command"):
            if st.session_state.ssh_client:
                command = process_commands[selected_cmd]
                output, error = execute_ssh_command(st.session_state.ssh_client, command)
                
                if output:
                    st.success("Command executed successfully:")
                    st.code(output, language="bash")
                if error:
                    st.error(f"Error: {error}")
            else:
                st.warning("Please establish SSH connection first")
    
    elif category == "Service Management":
        service_op = st.radio("Select Service Operation", [
            "Check Service Status", "Start Service", "Stop Service", "Restart Service", "List All Services"
        ])
        
        if service_op == "List All Services":
            if st.button("List Services"):
                if st.session_state.ssh_client:
                    output, error = execute_ssh_command(st.session_state.ssh_client, "systemctl list-units --type=service")
                    if output:
                        st.code(output, language="bash")
                    if error:
                        st.error(f"Error: {error}")
                else:
                    st.warning("Please establish SSH connection first")
        else:
            service_name = st.text_input("Service Name:")
            if st.button(f"{service_op}"):
                if service_name and st.session_state.ssh_client:
                    if service_op == "Check Service Status":
                        command = f"systemctl status {service_name}"
                    elif service_op == "Start Service":
                        command = f"systemctl start {service_name}"
                    elif service_op == "Stop Service":
                        command = f"systemctl stop {service_name}"
                    elif service_op == "Restart Service":
                        command = f"systemctl restart {service_name}"
                    
                    output, error = execute_ssh_command(st.session_state.ssh_client, command)
                    if output:
                        st.success("Command executed successfully:")
                        st.code(output, language="bash")
                    if error:
                        st.error(f"Error: {error}")
                else:
                    st.warning("Please provide service name and establish SSH connection")
    
    elif category == "File Operations":
        file_op = st.radio("Select File Operation", [
            "List Directory", "Create File", "Delete File", 
            "Create Directory", "Delete Directory", "View File Content"
        ])
        
        if file_op == "List Directory":
            directory = st.text_input("Directory Path:", value="~")
            if st.button("List Directory"):
                if st.session_state.ssh_client:
                    output, error = execute_ssh_command(st.session_state.ssh_client, f"ls -la {directory}")
                    if output:
                        st.code(output, language="bash")
                    if error:
                        st.error(f"Error: {error}")
                else:
                    st.warning("Please establish SSH connection first")
        
        elif file_op == "Create File":
            filename = st.text_input("Filename:")
            content = st.text_area("File Content (optional):")
            if st.button("Create File"):
                if filename and st.session_state.ssh_client:
                    if content:
                        command = f"echo '{content}' > {filename}"
                    else:
                        command = f"touch {filename}"
                    
                    output, error = execute_ssh_command(st.session_state.ssh_client, command)
                    if not error:
                        st.success(f"File {filename} created successfully")
                    else:
                        st.error(f"Error: {error}")
                else:
                    st.warning("Please provide filename and establish SSH connection")
        
        elif file_op == "Delete File":
            filename = st.text_input("Filename to delete:")
            if st.button("Delete File"):
                if filename and st.session_state.ssh_client:
                    output, error = execute_ssh_command(st.session_state.ssh_client, f"rm -f {filename}")
                    if not error:
                        st.success(f"File {filename} deleted successfully")
                    else:
                        st.error(f"Error: {error}")
                else:
                    st.warning("Please provide filename and establish SSH connection")
        
        elif file_op == "Create Directory":
            dirname = st.text_input("Directory name:")
            if st.button("Create Directory"):
                if dirname and st.session_state.ssh_client:
                    output, error = execute_ssh_command(st.session_state.ssh_client, f"mkdir -p {dirname}")
                    if not error:
                        st.success(f"Directory {dirname} created successfully")
                    else:
                        st.error(f"Error: {error}")
                else:
                    st.warning("Please provide directory name and establish SSH connection")
        
        elif file_op == "Delete Directory":
            dirname = st.text_input("Directory name to delete:")
            if st.button("Delete Directory"):
                if dirname and st.session_state.ssh_client:
                    output, error = execute_ssh_command(st.session_state.ssh_client, f"rm -rf {dirname}")
                    if not error:
                        st.success(f"Directory {dirname} deleted successfully")
                    else:
                        st.error(f"Error: {error}")
                else:
                    st.warning("Please provide directory name and establish SSH connection")
        
        elif file_op == "View File Content":
            filename = st.text_input("Filename to view:")
            if st.button("View File"):
                if filename and st.session_state.ssh_client:
                    output, error = execute_ssh_command(st.session_state.ssh_client, f"cat {filename}")
                    if output:
                        st.success(f"Content of {filename}:")
                        st.code(output, language="text")
                    if error:
                        st.error(f"Error: {error}")
                else:
                    st.warning("Please provide filename and establish SSH connection")
    
    elif category == "Custom Command":
        custom_cmd = st.text_input("Enter Custom Linux Command:")
        if st.button("Execute Custom Command"):
            if custom_cmd and st.session_state.ssh_client:
                output, error = execute_ssh_command(st.session_state.ssh_client, custom_cmd)
                
                if output:
                    st.success("Command executed successfully:")
                    st.code(output, language="bash")
                if error:
                    st.error(f"Error: {error}")
            else:
                st.warning("Please enter command and establish SSH connection")

elif section == "Docker":
    st.title("Docker Container Management")
    
    docker_task = st.selectbox("Select Docker Task", [
        "Container Operations",
        "Image Management", 
        "Apache Server Setup",
        "Multi-Container Setup"
    ])
    
    if docker_task == "Container Operations":
        st.subheader("Container Lifecycle Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("List Running Containers"):
                try:
                    result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.code(result.stdout, language="bash")
                    else:
                        st.error(f"Error: {result.stderr}")
                except Exception as e:
                    st.error(f"Docker command failed: {e}")
            
            if st.button("List All Containers"):
                try:
                    result = subprocess.run(['docker', 'ps', '-a'], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.code(result.stdout, language="bash")
                    else:
                        st.error(f"Error: {result.stderr}")
                except Exception as e:
                    st.error(f"Docker command failed: {e}")
        
        with col2:
            container_name = st.text_input("Container Name/ID:")
            
            if st.button("Start Container"):
                if container_name:
                    try:
                        result = subprocess.run(['docker', 'start', container_name], capture_output=True, text=True)
                        if result.returncode == 0:
                            st.success(f"Container {container_name} started successfully")
                        else:
                            st.error(f"Error: {result.stderr}")
                    except Exception as e:
                        st.error(f"Docker command failed: {e}")
                else:
                    st.warning("Please enter container name")
    
    elif docker_task == "Image Management":
        st.subheader("Docker Image Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("List Docker Images"):
                try:
                    result = subprocess.run(['docker', 'images'], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.code(result.stdout, language="bash")
                    else:
                        st.error(f"Error: {result.stderr}")
                except Exception as e:
                    st.error(f"Docker command failed: {e}")
            
            if st.button("Remove Unused Images"):
                try:
                    result = subprocess.run(['docker', 'image', 'prune', '-f'], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("Unused images removed successfully")
                        st.code(result.stdout, language="bash")
                    else:
                        st.error(f"Error: {result.stderr}")
                except Exception as e:
                    st.error(f"Docker command failed: {e}")
        
        with col2:
            image_name = st.text_input("Image Name (e.g., nginx:latest):")
            
            if st.button("Pull Image"):
                if image_name:
                    try:
                        with st.spinner(f"Pulling image {image_name}..."):
                            result = subprocess.run(['docker', 'pull', image_name], capture_output=True, text=True)
                            if result.returncode == 0:
                                st.success(f"Image {image_name} pulled successfully")
                            else:
                                st.error(f"Error: {result.stderr}")
                    except Exception as e:
                        st.error(f"Docker command failed: {e}")
                else:
                    st.warning("Please enter image name")
            
            if st.button("Remove Image"):
                if image_name:
                    try:
                        result = subprocess.run(['docker', 'rmi', image_name], capture_output=True, text=True)
                        if result.returncode == 0:
                            st.success(f"Image {image_name} removed successfully")
                        else:
                            st.error(f"Error: {result.stderr}")
                    except Exception as e:
                        st.error(f"Docker command failed: {e}")
                else:
                    st.warning("Please enter image name")
    
    elif docker_task == "Apache Server Setup":
        st.subheader("Apache Web Server in Docker")
        
        port = st.number_input("Host Port:", min_value=1000, max_value=65535, value=8080)
        container_name = st.text_input("Container Name:", value="apache-server")
        
        index_html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
        
        if os.path.exists(index_html_path):
            with open(index_html_path, 'r', encoding='utf-8') as f:
                default_html = f.read()
        else:
            default_html = """<!DOCTYPE html>
<html>
<head>
    <title>Apache Server</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        .container { max-width: 800px; margin: 0 auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Welcome to Apache Server</h1>
        <p>This is a custom Apache server running in a Docker container.</p>
        <p>Edit this HTML in the text area below to customize your page.</p>
    </div>
</body>
</html>"""
        
        st.subheader("Edit HTML Content")
        custom_html = st.text_area("Modify the HTML content below:", value=default_html, height=400)
        
        if st.button("Update and Start Apache Server"):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    html_dir = os.path.join(temp_dir, "html")
                    os.makedirs(html_dir, exist_ok=True)
                    
                    with open(os.path.join(html_dir, "index.html"), "w", encoding='utf-8') as f:
                        f.write(custom_html)
                    
                    dockerfile = os.path.join(temp_dir, "Dockerfile")
                    dockerfile_content = """
FROM httpd:2.4

COPY html/ /usr/local/apache2/htdocs/

EXPOSE 80
"""
                    with open(dockerfile, "w") as f:
                        f.write(dockerfile_content)
                    
                    stop_cmd = f"docker stop {container_name} 2>nul || true"
                    remove_cmd = f"docker rm {container_name} 2>nul || true"
                    subprocess.run(stop_cmd, shell=True)
                    subprocess.run(remove_cmd, shell=True)
                    
                    build_cmd = f"docker build -t {container_name}-image {temp_dir}"
                    build_result = subprocess.run(build_cmd, shell=True, capture_output=True, text=True)
                    
                    if build_result.returncode != 0:
                        st.error(f"Failed to build Docker image: {build_result.stderr}")
                        st.stop()
                    
                    run_cmd = f"docker run -d -p {port}:80 --name {container_name} {container_name}-image"
                    run_result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)
                    
                    if run_result.returncode != 0:
                        st.error(f"Failed to run Docker container: {run_result.stderr}")
                        st.stop()
                    
                    st.success(f"Apache server is now running at http://localhost:{port}")
                    st.info(f"You can access the web interface at: http://localhost:{port}")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    elif docker_task == "Multi-Container Setup":
        st.subheader("Multi-Container Application Setup")
        
        setup_type = st.radio("Select Setup Type", [
            "LAMP Stack (Linux, Apache, MySQL, PHP)",
            "MEAN Stack (MongoDB, Express, Angular, Node.js)",
            "WordPress with MySQL",
            "Custom Docker Compose"
        ])
        
        if setup_type == "LAMP Stack (Linux, Apache, MySQL, PHP)":
            port = st.number_input("Host Port:", min_value=1000, max_value=65535, value=8080)
            container_name = st.text_input("Container Name:", value="lamp-stack")
            password=st.text_input("Password:", type="password")
            db_name=st.text_input("Database Name:") 
            if st.button("Deploy LAMP Stack"):
                compose_content = f"""version: '3.8'
services:
  web:
    image: php:7.4-apache
    ports:
      - "{port}:80"
    volumes:
      - ./html:/var/www/html
    depends_on:
      - db
  db:
    image: mysql:5.7
    environment:
      MYSQL_ROOT_PASSWORD: {password}
      MYSQL_DATABASE: {db_name}
    ports:
      - "3306:3306"
  phpmyadmin:
    image: phpmyadmin/phpmyadmin
    ports:
      - "8081:80"
    environment:
      PMA_HOST: db
    depends_on:
      - db"""
                
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        compose_file = os.path.join(temp_dir, "docker-compose.yml")
                        html_dir = os.path.join(temp_dir, "html")
                        os.makedirs(html_dir, exist_ok=True)
                        
                        with open(compose_file, "w") as f:
                            f.write(compose_content)
                        
                        index_php = """<?php
echo "<h1>LAMP Stack is Working!</h1>";
echo "<p>PHP Version: " . phpversion() . "</p>";
echo "<p>Server Time: " . date('Y-m-d H:i:s') . "</p>";
?>"""
                        
                        with open(os.path.join(html_dir, "index.php"), "w") as f:
                            f.write(index_php)
                        
                        with st.spinner("Deploying LAMP stack..."):
                            result = subprocess.run([
                                'docker-compose', '-f', compose_file, 'up', '-d'
                            ], capture_output=True, text=True, cwd=temp_dir)
                            
                            if result.returncode == 0:
                                st.success("LAMP stack deployed successfully!")
                                st.info("Access points:")
                                st.write("- Web Server: http://localhost:8080")
                                st.write("- phpMyAdmin: http://localhost:8081")
                            else:
                                st.error(f"Error: {result.stderr}")
                except Exception as e:
                    st.error(f"Error deploying LAMP stack: {e}")
        
        elif setup_type == "WordPress with MySQL":
            port = st.number_input("Host Port:", min_value=1000, max_value=65535, value=8080)
            container_name = st.text_input("Container Name:", value="wordpress")
            root_password=st.text_input("ROOT Password:", type="password")
            user=st.text_input("Wordpress User:")
            password=st.text_input("Wordpress Password:", type="password")
            db_name=st.text_input("Wordpress Database Name:")
            mysql_user=st.text_input("MySQL User:")
            mysql_password=st.text_input("MySQL Password:", type="password")
            mysql_db_name=st.text_input("MySQL Database Name:") 
            if st.button("Deploy WordPress"):
                compose_content = f"""version: '3.8'
services:
  wordpress:
    image: wordpress:latest
    ports:
      - "{port}:80"
    environment:
      WORDPRESS_DB_HOST: db:3306
      WORDPRESS_DB_USER: {user}
      WORDPRESS_DB_PASSWORD: {password}
      WORDPRESS_DB_NAME: {db_name}
    depends_on:
      - db
  db:
    image: mysql:5.7
    environment:
      MYSQL_DATABASE: {mysql_db_name}
      MYSQL_USER: {mysql_user}
      MYSQL_PASSWORD: {mysql_password}
      MYSQL_ROOT_PASSWORD: {root_password}"""
                
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        compose_file = os.path.join(temp_dir, "docker-compose.yml")
                        
                        with open(compose_file, "w") as f:
                            f.write(compose_content)
                        
                        with st.spinner("Deploying WordPress..."):
                            result = subprocess.run([
                                'docker-compose', '-f', compose_file, 'up', '-d'
                            ], capture_output=True, text=True, cwd=temp_dir)
                            
                            if result.returncode == 0:
                                st.success("WordPress deployed successfully!")
                                st.info("Access WordPress at: http://localhost:8080")
                            else:
                                st.error(f"Error: {result.stderr}")
                except Exception as e:
                    st.error(f"Error deploying WordPress: {e}")
        
        elif setup_type == "Custom Docker Compose":
            port = st.number_input("Host Port:", min_value=1000, max_value=65535, value=8080)
            container_name = st.text_input("Container Name:", value="custom-compose")
            compose_yaml = st.text_area("Docker Compose YAML:", height=300, value=f"""version: '3.8'
services:
  web:
    image: nginx:latest
    ports:
      - "{port}:80"
  app:
    image: node:latest
    command: node -e "console.log('Hello from Node.js')"
""")
            
            if st.button("Deploy Custom Compose"):
                if compose_yaml:
                    try:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            compose_file = os.path.join(temp_dir, "docker-compose.yml")
                            
                            with open(compose_file, "w") as f:
                                f.write(compose_yaml)
                            
                            with st.spinner("Deploying custom compose..."):
                                result = subprocess.run([
                                    'docker-compose', '-f', compose_file, 'up', '-d'
                                ], capture_output=True, text=True, cwd=temp_dir)
                                
                                if result.returncode == 0:
                                    st.success("Custom compose deployed successfully!")
                                    st.code(result.stdout, language="bash")
                                else:
                                    st.error(f"Error: {result.stderr}")
                    except Exception as e:
                        st.error(f"Error deploying custom compose: {e}")
                else:
                    st.warning("Please provide Docker Compose YAML")

elif section == "AWS":
    st.title("AWS Cloud Services")
    
    aws_service = st.selectbox("Select AWS Service", [
        "EC2 Management",
        "CloudWatch Logs", 
        "S3 Operations",
        "Lambda Functions",
        "Audio Transcription",
        "MongoDB Integration"
    ])
    
    st.subheader("AWS Credentials")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        aws_access_key = st.text_input("AWS Access Key ID:", type="password")
    with col2:
        aws_secret_key = st.text_input("AWS Secret Access Key:", type="password")
    with col3:
        aws_region = st.selectbox("AWS Region:", [
            "us-east-1", "us-west-2", "eu-west-1", "ap-south-1", "ap-southeast-1"
        ])
    
    if aws_service == "EC2 Management":
        st.subheader("EC2 Instance Operations")
        
        if aws_access_key and aws_secret_key:
            try:
                ec2 = boto3.client('ec2', 
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=aws_region
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("List EC2 Instances"):
                        try:
                            response = ec2.describe_instances()
                            instances = []
                            for reservation in response['Reservations']:
                                for instance in reservation['Instances']:
                                    instances.append({
                                        'Instance ID': instance['InstanceId'],
                                        'State': instance['State']['Name'],
                                        'Type': instance['InstanceType'],
                                        'Launch Time': str(instance['LaunchTime'])
                                    })
                            
                            if instances:
                                st.json(instances)
                            else:
                                st.info("No EC2 instances found")
                        except Exception as e:
                            st.error(f"Error listing instances: {e}")
                
                with col2:
                    instance_id = st.text_input("Instance ID:")
                    
                    col2_1, col2_2 = st.columns(2)
                    with col2_1:
                        if st.button("Start Instance"):
                            if instance_id:
                                try:
                                    ec2.start_instances(InstanceIds=[instance_id])
                                    st.success(f"Instance {instance_id} start initiated")
                                except Exception as e:
                                    st.error(f"Error starting instance: {e}")
                    
                    with col2_2:
                        if st.button("Stop Instance"):
                            if instance_id:
                                try:
                                    ec2.stop_instances(InstanceIds=[instance_id])
                                    st.success(f"Instance {instance_id} stop initiated")
                                except Exception as e:
                                    st.error(f"Error stopping instance: {e}")
            except Exception as e:
                st.error(f"AWS connection failed: {e}")
        else:
            st.warning("Please provide AWS credentials")
    
    elif aws_service == "CloudWatch Logs":
        st.subheader("CloudWatch Logs Access")
        
        if aws_access_key and aws_secret_key:
            try:
                logs = boto3.client('logs',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=aws_region
                )
                
                if st.button("List Log Groups"):
                    try:
                        response = logs.describe_log_groups()
                        log_groups = [lg['logGroupName'] for lg in response['logGroups']]
                        if log_groups:
                            st.write("Available log groups:")
                            for lg in log_groups[:10]:
                                st.write(f"- {lg}")
                        else:
                            st.info("No log groups found")
                    except Exception as e:
                        st.error(f"Error listing log groups: {e}")
                
                log_group_name = st.text_input("Log Group Name:")
                if st.button("Get Recent Logs"):
                    if log_group_name:
                        try:
                            response = logs.describe_log_streams(logGroupName=log_group_name)
                            if response['logStreams']:
                                stream_name = response['logStreams'][0]['logStreamName']
                                events = logs.get_log_events(
                                    logGroupName=log_group_name,
                                    logStreamName=stream_name,
                                    limit=50
                                )
                                
                                st.write(f"Recent logs from {log_group_name}:")
                                for event in events['events']:
                                    st.text(f"{event['timestamp']}: {event['message']}")
                            else:
                                st.info("No log streams found")
                        except Exception as e:
                            st.error(f"Error getting logs: {e}")
                    else:
                        st.warning("Please provide log group name")
            except Exception as e:
                st.error(f"AWS CloudWatch connection failed: {e}")
        else:
            st.warning("Please provide AWS credentials")
    
    elif aws_service == "S3 Operations":
        st.subheader("S3 Bucket Operations")
        
        if aws_access_key and aws_secret_key:
            try:
                s3 = boto3.client('s3',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=aws_region
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("List S3 Buckets"):
                        try:
                            response = s3.list_buckets()
                            buckets = [bucket['Name'] for bucket in response['Buckets']]
                            if buckets:
                                st.write("Available buckets:")
                                for bucket in buckets:
                                    st.write(f"- {bucket}")
                            else:
                                st.info("No S3 buckets found")
                        except Exception as e:
                            st.error(f"Error listing buckets: {e}")
                    
                    bucket_name = st.text_input("Bucket Name:")
                    if st.button("Create Bucket"):
                        if bucket_name:
                            try:
                                s3.create_bucket(Bucket=bucket_name)
                                st.success(f"Bucket {bucket_name} created successfully")
                            except Exception as e:
                                st.error(f"Error creating bucket: {e}")
                        else:
                            st.warning("Please provide bucket name")
                
                with col2:
                    uploaded_file = st.file_uploader("Choose file to upload:")
                    upload_bucket = st.text_input("Upload to Bucket:")
                    
                    if st.button("Upload to S3"):
                        if upload_bucket and uploaded_file:
                            try:
                                s3.upload_fileobj(uploaded_file, upload_bucket, uploaded_file.name)
                                st.success(f"File {uploaded_file.name} uploaded to {upload_bucket}")
                            except Exception as e:
                                st.error(f"Error uploading file: {e}")
                        else:
                            st.warning("Please provide bucket name and select a file")
                    
                    if st.button("Upload Without Login (Public)"):
                        st.info("For public uploads, use pre-signed URLs or IAM roles")
                        if upload_bucket and uploaded_file:
                            try:
                                presigned_url = s3.generate_presigned_url(
                                    'put_object',
                                    Params={'Bucket': upload_bucket, 'Key': uploaded_file.name},
                                    ExpiresIn=3600
                                )
                                st.code(f"Pre-signed URL: {presigned_url}")
                            except Exception as e:
                                st.error(f"Error generating pre-signed URL: {e}")
            except Exception as e:
                st.error(f"AWS S3 connection failed: {e}")
        else:
            st.warning("Please provide AWS credentials")
    
    elif aws_service == "Lambda Functions":
        st.subheader("Lambda Function Integration")
        
        if aws_access_key and aws_secret_key:
            try:
                lambda_client = boto3.client('lambda',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=aws_region
                )
                
                if st.button("List Lambda Functions"):
                    try:
                        response = lambda_client.list_functions()
                        functions = [f['FunctionName'] for f in response['Functions']]
                        if functions:
                            st.write("Available Lambda functions:")
                            for func in functions:
                                st.write(f"- {func}")
                        else:
                            st.info("No Lambda functions found")
                    except Exception as e:
                        st.error(f"Error listing functions: {e}")
                
                function_name = st.text_input("Function Name:")
                payload = st.text_area("Payload (JSON):", value='{"key": "value"}')
                
                if st.button("Invoke Lambda Function"):
                    if function_name:
                        try:
                            response = lambda_client.invoke(
                                FunctionName=function_name,
                                Payload=payload
                            )
                            result = response['Payload'].read().decode()
                            st.success("Function invoked successfully:")
                            st.code(result, language="json")
                        except Exception as e:
                            st.error(f"Error invoking function: {e}")
                    else:
                        st.warning("Please provide function name")
            except Exception as e:
                st.error(f"AWS Lambda connection failed: {e}")
        else:
            st.warning("Please provide AWS credentials")
    
    elif aws_service == "Audio Transcription":
        st.subheader("AWS Transcribe - Audio to Text")
        
        if aws_access_key and aws_secret_key:
            try:
                transcribe = boto3.client('transcribe',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=aws_region
                )
                
                s3 = boto3.client('s3',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=aws_region
                )
                
                uploaded_audio = st.file_uploader("Choose audio file:", type=['mp3', 'wav', 'm4a'])
                bucket_name = st.text_input("S3 Bucket for Audio:")
                job_name = st.text_input("Transcription Job Name:", value="audio-transcription-job")
                
                if st.button("Start Transcription"):
                    if uploaded_audio and bucket_name and job_name:
                        try:
                            with st.spinner("Uploading audio to S3..."):
                                audio_key = f"audio/{uploaded_audio.name}"
                                s3.upload_fileobj(uploaded_audio, bucket_name, audio_key)
                                
                            with st.spinner("Starting transcription job..."):
                                transcribe.start_transcription_job(
                                    TranscriptionJobName=job_name,
                                    Media={'MediaFileUri': f's3://{bucket_name}/{audio_key}'},
                                    MediaFormat=uploaded_audio.name.split('.')[-1],
                                    LanguageCode='en-US'
                                )
                                
                            st.success("Transcription job started successfully!")
                            st.info("Check job status below")
                        except Exception as e:
                            st.error(f"Error starting transcription: {e}")
                    else:
                        st.warning("Please provide all required fields")
                
                if st.button("Check Transcription Status"):
                    if job_name:
                        try:
                            response = transcribe.get_transcription_job(TranscriptionJobName=job_name)
                            status = response['TranscriptionJob']['TranscriptionJobStatus']
                            st.write(f"Job Status: {status}")
                            
                            if status == 'COMPLETED':
                                transcript_uri = response['TranscriptionJob']['Transcript']['TranscriptFileUri']
                                st.success("Transcription completed!")
                                st.write(f"Transcript URI: {transcript_uri}")
                        except Exception as e:
                            st.error(f"Error checking status: {e}")
                    else:
                        st.warning("Please provide job name")
            except Exception as e:
                st.error(f"AWS Transcribe connection failed: {e}")
        else:
            st.warning("Please provide AWS credentials")
    
    elif aws_service == "MongoDB Integration":
        st.subheader("MongoDB Service Integration")
        
        st.info("This section demonstrates connecting to MongoDB via Lambda or DocumentDB")
        
        if aws_access_key and aws_secret_key:
            connection_string = st.text_input("MongoDB Connection String:", 
                                            value="mongodb://username:password@cluster.amazonaws.com:27017/database")
            database_name = st.text_input("Database Name:", value="mydb")
            collection_name = st.text_input("Collection Name:", value="mycollection")
            
            if st.button("Test MongoDB Connection"):
                st.code(f"""
# Lambda function code for MongoDB connection
import pymongo
import json

def lambda_handler(event, context):
    try:
        client = pymongo.MongoClient('{connection_string}')
        db = client['{database_name}']
        collection = db['{collection_name}']
        
        # Test connection
        result = collection.find_one()
        
        return {{
            'statusCode': 200,
            'body': json.dumps('MongoDB connection successful!')
        }}
    except Exception as e:
        return {{
            'statusCode': 500,
            'body': json.dumps(f'Error: {{str(e)}}')
        }}
                """, language="python")
                st.info("Deploy this code as a Lambda function to test MongoDB connectivity")
        else:
            st.warning("Please provide AWS credentials")

elif section == "Projects":
    st.title("Automation Projects")
    
    project = st.selectbox("Select Project", [
        "Gemini Query resolving Agent",
        "Cloud Automation using Python", 
        "Running Apache Inside Docker Container",
        "CI/CD from Scratch: Flask + Jenkins + Docker"
    ])
    
    if project == "Gemini Query resolving Agent":
        st.subheader("Gemini Query resolving Agent")
        st.write("A smart agent built using Gemini to respond dynamically to user queries.")
        
        user_query = st.text_input("Enter your query:")
        if st.button("Process with Gemini Agent"):
            if user_query:
                model = OpenAI(api_key="Your_API_key_here.", base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
                msg = [
                    {"role": "system", "content": "You are a very skilled and an all rounder person with a high level knowledge of each and every techs and other stuff.The repit bigonse you generate to the query has to be 4 liner at most and bigger if the person needs "},
                    {"role": "user", "content": f"Generate the answer according to {user_query}"}
                ]
                res = model.chat.completions.create(model="gemini-2.5-flash", messages=msg)
                st.write(str(res.choices[0].message.content))
                
            else:
                st.warning("Please enter a query")
    
    elif project == "Cloud Automation using Python":
        try:
            ec2_client = boto3.client('ec2')
            ec2_resource = boto3.resource('ec2')
            asg_client = boto3.client('autoscaling')
            ec2_client.describe_regions()
        except ClientError as e:
            st.error(f"AWS credentials error: {e}. Please ensure your AWS credentials are set up correctly.")
        except Exception as e:
            st.error(f"An unexpected error occurred during AWS client initialization: {e}")
            st.stop()
        st.subheader("Cloud Automation using Python")
        st.write("Automating cloud infrastructure tasks such as provisioning and monitoring using Python and relevant SDKs like Boto3.")
        
        automation_task = st.radio("Select Automation Task", [
            "Infrastructure Provisioning",
            "Resource Monitoring", 
            "Auto Scaling Configuration",
            "Backup Automation"
        ])
        
        if st.button("Execute Cloud Automation"):
            if automation_task=="Infrastucture Provisioning":
                st.info("Provisioning a new EC2 instance...")
                try:
                    instance = ec2_resource.create_instances(
                        ImageId=st.text_input("Enter the Ami ID for the prefered OS:"),
                        InstanceType=st.text_input("Enter the Instance type:"),
                        MinCount=1,
                        MaxCount=1,
                        KeyName=st.text_input("Enter the key pair name(Existing):"),
                        TagSpecifications=[
                            {
                                'ResourceType': 'instance',
                                'Tags': [{'Key': 'Name', 'Value': 'Streamlit-Automated-Instance'}]
                            },
                        ]
                    )
                    st.success(f"✅ Infrastructure provisioning request sent! Instance ID: {instance[0].id}")
                except ClientError as e:
                    st.error(f"Error provisioning instance: {e}")

            if automation_task=="Resource Monitoring":
                st.info("Checking resource statuses...")
                try:
                    response = ec2_client.describe_instances(Filters=[{'Name': 'instance-state-name', 'Values': ['running', 'pending', 'stopped']}])
                    instances = []
                    for reservation in response['Reservations']:
                        for instance in reservation['Instances']:
                            instance_id = instance['InstanceId']
                            state = instance['State']['Name']
                            instances.append(f"• Instance ID: {instance_id}, State: {state}")
                    
                    if instances:
                        st.success("✅ Resource monitoring complete! Here are the current instances:")
                        for instance_info in instances:
                            st.write(instance_info)
                    else:
                        st.warning("⚠️ No running or pending instances found.")
                except ClientError as e:
                    st.error(f"Error monitoring resources: {e}")

            if automation_task=="Auto Scaling Configuration":
                st.info("Configuring auto-scaling group...")
                try:
                    asg_name = st.text_input("Enter a name for the Auto Scaling Group:")
                    
                    response = asg_client.create_launch_configuration(
                        LaunchConfigurationName='Streamlit-Launch-Config',
                        ImageId=st.text_input("Enter a valid Ami ID:"),
                        InstanceType='t2.micro',
                    )
                    st.info("Launch configuration created.")
                    
                    response = asg_client.create_auto_scaling_group(
                        AutoScalingGroupName=asg_name,
                        LaunchConfigurationName='Streamlit-Launch-Config',
                        MinSize=1,
                        MaxSize=3,
                        DesiredCapacity=1,
                        AvailabilityZones=['ap-south-1a','ap-south-1b','ap-south-1c'],
                        Tags=[{'Key': 'Name', 'Value': asg_name}]
                    )
                    st.success("✅ Auto-scaling configuration complete! The group is now set up.")
                except ClientError as e:
                    st.error(f"Error configuring auto-scaling: {e}")        
            
            else:
                st.info("Initiating database backup...")
                try:
                    volume_id = st.text_input("Enter a valid volumne id for which you need the snapshot for:")
                    response = ec2_client.create_snapshot(
                        VolumeId=volume_id,
                        Description='Automated backup via Streamlit app',
                        TagSpecifications=[
                            {
                                'ResourceType': 'snapshot',
                                'Tags': [{'Key': 'Name', 'Value': f"Backup-{time.strftime('%Y-%m-%d-%H-%M-%S')}"}]
                            },
                        ]
                    )
                    st.success(f"✅ Backup automation complete! Snapshot ID: {response['SnapshotId']}")
                except ClientError as e:
                    st.error(f"Error creating backup: {e}")
            st.info(f"Executing {automation_task} automation task")
    
    elif project == "Running Apache Inside Docker Container":
        st.subheader("Apache in Docker Container")
        st.write("Deploying and managing an Apache web server inside a Docker container for isolated and portable web hosting.")
        
        port = st.number_input("Port for Apache Server:", min_value=1000, max_value=65535, value=8080)
        container_name = st.text_input("Container Name:", value="apache-web-server")
        
        if st.button("Deploy Apache Container"):
            try:
                with st.spinner("Deploying Apache container..."):
                    subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True)
                    
                    result = subprocess.run([
                        'docker', 'run', '-d', 
                        '--name', container_name,
                        '-p', f'{port}:80',
                        'httpd:latest'
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        st.success("Apache container deployed successfully!")
                        st.info(f"🌐 Access your Apache server at: http://localhost:{port}")
                        st.code(f"Container ID: {result.stdout.strip()}", language="text")
                        
                        status_result = subprocess.run([
                            'docker', 'ps', '--filter', f'name={container_name}', '--format', 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
                        ], capture_output=True, text=True)
                        
                        if status_result.returncode == 0:
                            st.subheader("Container Status:")
                            st.code(status_result.stdout, language="bash")
                    else:
                        st.error(f"Error deploying container: {result.stderr}")
            except Exception as e:
                st.error(f"Docker command failed: {e}")
        
        if st.button("Deploy Custom Apache with HTML"):
            custom_html = st.text_area("Custom HTML Content:", value="""<!DOCTYPE html>
<html>
<head>
    <title>Apache in Docker - Project Demo</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f4f4f4; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; }
        .info { background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐳 Apache Server Running in Docker</h1>
        <div class="info">
            <h3>Project: Running Apache Inside Docker Container</h3>
            <p>This Apache web server is running inside a Docker container, providing:</p>
            <ul>
                <li>✅ Isolated and portable web hosting</li>
                <li>✅ Easy deployment and scaling</li>
                <li>✅ Consistent environment across platforms</li>
                <li>✅ Automated deployment from dashboard</li>
            </ul>
        </div>
        <p><strong>Deployment Status:</strong> Successfully deployed from Centralized Dashboard!</p>
        <p><strong>Container Technology:</strong> Docker with Apache HTTP Server</p>
        <p><strong>Access Time:</strong> <span id="datetime"></span></p>
    </div>
    <script>
        document.getElementById('datetime').innerHTML = new Date().toLocaleString();
    </script>
</body>
</html>""")
            
            if st.button("Deploy Custom Apache"):
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        html_file = os.path.join(temp_dir, "index.html")
                        dockerfile = os.path.join(temp_dir, "Dockerfile")
                        
                        with open(html_file, "w") as f:
                            f.write(custom_html)
                        
                        dockerfile_content = """FROM httpd:latest
COPY index.html /usr/local/apache2/htdocs/
EXPOSE 80"""
                        
                        with open(dockerfile, "w") as f:
                            f.write(dockerfile_content)
                        
                        with st.spinner("Building and deploying custom Apache..."):
                            subprocess.run(['docker', 'rm', '-f', f'{container_name}-custom'], capture_output=True)
                            
                            build_result = subprocess.run([
                                'docker', 'build', '-t', f'{container_name}-custom', temp_dir
                            ], capture_output=True, text=True)
                            
                            if build_result.returncode == 0:
                                run_result = subprocess.run([
                                    'docker', 'run', '-d', 
                                    '--name', f'{container_name}-custom',
                                    '-p', f'{port}:80',
                                    f'{container_name}-custom'
                                ], capture_output=True, text=True)
                                
                                if run_result.returncode == 0:
                                    st.success("Custom Apache server deployed successfully!")
                                    st.info(f"🌐 Access your custom server at: http://localhost:{port}")
                                    st.code(f"Container ID: {run_result.stdout.strip()}", language="text")
                                else:
                                    st.error(f"Error running container: {run_result.stderr}")
                            else:
                                st.error(f"Error building image: {build_result.stderr}")
                except Exception as e:
                    st.error(f"Error creating custom Apache: {e}")
        
        st.subheader("Container Management")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Stop Container"):
                try:
                    result = subprocess.run(['docker', 'stop', container_name], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success(f"Container {container_name} stopped")
                    else:
                        st.error(f"Error: {result.stderr}")
                except Exception as e:
                    st.error(f"Error stopping container: {e}")
        
        with col2:
            if st.button("Start Container"):
                try:
                    result = subprocess.run(['docker', 'start', container_name], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success(f"Container {container_name} started")
                    else:
                        st.error(f"Error: {result.stderr}")
                except Exception as e:
                    st.error(f"Error starting container: {e}")
        
        with col3:
            if st.button("Remove Container"):
                try:
                    result = subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success(f"Container {container_name} removed")
                    else:
                        st.error(f"Error: {result.stderr}")
                except Exception as e:
                    st.error(f"Error removing container: {e}")
    
    elif project == "CI/CD from Scratch: Flask + Jenkins + Docker":
        st.subheader("Flask CI/CD Pipeline")
        st.write("Building a complete CI/CD pipeline using Jenkins to deploy a Flask app inside Docker containers.")
        
        pipeline_stage = st.selectbox("Select Pipeline Stage", [
            "Source Code Management",
            "Build Stage",
            "Test Stage", 
            "Docker Image Creation",
            "Deployment Stage"
        ])
        
        if st.button("Generate Pipeline Configuration"):
            st.code(f"""
# Jenkins Pipeline Stage: {pipeline_stage}
pipeline {{
    agent any
    stages {{
        stage('{pipeline_stage}') {{
            steps {{
                echo 'Executing {pipeline_stage}'
            }}
        }}
    }}
}}
            """, language="groovy")
            st.info(f"Pipeline configuration for {pipeline_stage} generated")
