Since the application is dockerize, it doesnt needs to install any requirements.txt file

# MongoDB Installation

Ensure MongoDB is installed on your system.

## Installing MongoDB
### On Linux (Ubuntu)
```bash
sudo apt update
sudo apt install -y mongodb
sudo systemctl start mongodb
sudo systemctl enable mongodb
```
### On Windows (Using WSL)
```bash
sudo apt update
sudo apt install -y mongodb
sudo service mongodb start
```

# Kubernetes for Scalability

## Installing Kubernetes on WSL (Windows) or Linux
### On WSL (Windows)
```bash
wsl --install -d Ubuntu
curl -LO "https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl"
sudo chmod +x kubectl
sudo mv kubectl /usr/local/bin/
kubectl version --client
```
### On Linux
```bash
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt update
sudo apt install -y kubectl
kubectl version --client
```

# Running Minikube Cluster

Start the Minikube cluster:
```bash
minikube start
```

Run the frontend service in another terminal:
```bash
kubectl apply -f frontend-deployment.yml
```

Apply the config file to enable communication between Kubernetes and MongoDB:
```bash
kubectl apply -f frontend-configmap.yml
```

Apply the Horzintal Pod Scaling For Autoscaling and Self-Healing
```bash
kubectl apply -f frontend-hpa.yml
```

## Port Forwarding for Local Access
To give access to your local machine for these services, apply port forwarding:
```bash
kubectl port-forward svc/frontend-service 5000:5000
```

## (Optional) Run individual pods:
- First, get the pod names:
```bash
kubectl get pods
```
- Then, forward specific pods to different ports:
```bash
kubectl port-forward pods/frontend-deployment<pod_name> 5001:5000
kubectl port-forward pods/frontend-deployment<pod_name> 5002:8000
```

Check if the pods and deployments have been created:
```bash
kubectl get pods
kubectl get deployments
```

## Accessing the Application
Open your browser and navigate to:
```bash
localhost:5000
```

## Monitoring Kubernetes
To monitor pod health and API status, use Minikube Dashboard:
```bash
minikube dashboard
```

