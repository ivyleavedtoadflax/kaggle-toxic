variable "aws_region" { default = "eu-west-1" } # Ireland
variable "username" { default = "databoxuser"}
variable "instance_type" { default = "t2.micro" }
variable "volume_size" { default = "40" }
variable "profile" { default = "deeplearning" }
variable "create_snapshot" { default = "" }
variable "local_ip" { default = "0.0.0.0/0" }
variable "snapshot_id" {
  default = ""
  description = "Specify a snapshot_id to be used when creating the EBS Volume. Note that this snapshot must be in the same region as the instance, and must be the same size or smaller than the volume as specified in volume_size."
}

variable "public_key_path" {
  description = "Enter the path to the SSH Public Key to add to AWS."
  default = "~/.ssh/id_rsa.pub"
}

variable "ami_id" {
  default     = "" # Note this is empty.
  description = "Use this specific AMI ID for our EC2 instance. Default is Ubuntu 16.04 LTS in the current region"
}

provider "aws" {
    region = "${var.aws_region}"
    profile = "${var.profile}"
}


resource "aws_security_group" "allow_ssh_only" {
  name_prefix = "DataBox-SecurityGroup-SSH"
  description = "Allow inbound SSH traffic from SSH only"

  ingress = {
    from_port = 22
    to_port = 22
    protocol = "tcp"
    cidr_blocks = ["${var.local_ip}"]
  }
}

resource "aws_security_group" "allow_all_outbound" {
  name_prefix = "DataBox-SecurityGroup-outbound"
  description = "Allow all outbound traffic"

  egress = {
    from_port = 0
    to_port = 0
    protocol = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

data "aws_ami" "ubuntu" {
    most_recent = true

    filter {
        name   = "name" 
        values = ["ubuntu/images/hvm-ssd/ubuntu-xenial-16.04-amd64-server-*"]
    }

    filter {
        name   = "virtualization-type"
        values = ["hvm"]
    }

    owners = ["099720109477"] # Canonical
}

resource "aws_instance" "box" {
    ami = "${var.ami_id != "" ? var.ami_id : data.aws_ami.ubuntu.id}"
    availability_zone = "${var.aws_region}a"
    instance_type = "${var.instance_type}"
    security_groups = [
        "${aws_security_group.allow_ssh_only.name}",
        "${aws_security_group.allow_all_outbound.name}"
        ]
    key_name = "${aws_key_pair.auth.key_name}"

    provisioner "remote-exec" {
        inline = [
            "sudo locale-gen en_GB.UTF-8",
            "sudo apt-get -y update",
            "sudo apt-get -y install python-dev"
        ]

        connection {
            type        = "ssh"
            user        = "ubuntu"
            agent       = true
        }
    }

    tags {
        Name = "DataBoxEC2"
    }
}

resource "aws_ebs_volume" "volume" {
    availability_zone = "${var.aws_region}a"
    size = "${var.volume_size}"
    snapshot_id = "${var.snapshot_id}"
    tags {
        Name = "DataBoxVolume"
    }

    provisioner "local-exec" {
	when = "destroy"
	command = "if [ ${var.create_snapshot} ]; then aws ec2 create-snapshot --volume-id ${aws_ebs_volume.volume.id} --region ${var.aws_region}  --profile ${var.profile} ; fi;"
	on_failure = "fail"
    }
}

resource "aws_volume_attachment" "ebs_att" {
  device_name = "/dev/xvdh"
  volume_id   = "${aws_ebs_volume.volume.id}"
  instance_id = "${aws_instance.box.id}"
}

resource "aws_key_pair" "auth" {
  key_name   = "databox-key-${var.username}"
  public_key = "${file(var.public_key_path)}"
}

output "ec2_ip" {
    value = "${aws_instance.box.public_ip}"
}
output "data_volume_id" {
    value = "${aws_ebs_volume.volume.id}"
}
