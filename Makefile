
ip:
	@echo ${LOCAL_IP}

local_run: 
	python3 src/GRU.py

connect:
	ssh -L localhost:6006:localhost:6006 ubuntu@`terraform output ec2_ip`

up:
	./databox.sh -i ${INSTANCE} -a ${AMI} \
	-r ${REGION} -p ${PLAYBOOK} \
	-s ${SNAPSHOT_ID} -l ${LOCAL_IP} up

down:
	./databox.sh -r ${REGION} down

create_snapshot:
		aws ec2 create-snapshot --volume-id ${VOLUME_ID} \
		--region ${REGION} --profile ${PROFILE} \
		--description "${DESCRIPTION}"

describe_snapshots:
		aws ec2 describe-snapshots \
		--region ${REGION} --profile ${PROFILE} \
		--owner-id ${OWNER_ID}

describe_instances:
		aws ec2 describe-instances \
		--region ${REGION} --profile ${PROFILE}
.PHONY:

