
if __name__ == "__main__":
	gpu="0"
	gamma=0.99 
	num_steps="400"

	for pos in ["head"]:
		for lr in ["1e-4"]:
			for tol in [0.5]:
				for wheel_vel in [0.2]:
					for arm_vel in [0.0]:
						for suc_rwd in [30.0]:
							for pot_rwd in [5.0]:
								for col_rwd in [-10.0]:

									file_name = "jr2_{}_tol_{}_suc_rwd_{}_pot_rwd_{}_col_rwd_{}_gma_{}_lr_{}_nstps_{}_spd_{}_{}.yaml".format(pos,tol,suc_rwd,pot_rwd,col_rwd,gamma,lr,num_steps,wheel_vel,arm_vel)
									
									f = open(file_name, 'w').close()
									f = open(file_name, 'w')
									
									f.write("scene: stadium\n\n")

									f.write("robot: JR2_Kinova_Head\n")
									f.write("wheel_velocity: {}\n".format(wheel_vel))
									f.write("head_velocity: 0.2\n")
									f.write("arm_velocity: {}\n\n".format(arm_vel))

									f.write("task: pointgoal\n")
									f.write("fisheye: false\n\n")

									f.write("initial_pos: [0.0, 0.0, 0.0]\n")
									f.write("initial_orn: [0.0, 0.0, 0.0]\n\n")

									f.write("target_pos: [4.0, 0.0, 0.0]\n")
									f.write("target_orn: [0.0, 0.0, 0.0]\n\n")

									f.write("is_discrete: false\n")
									f.write("additional_states_dim: 4\n")
									f.write("goal_dim: 0\n\n")

									f.write("reward_type: l2\n")
									f.write("success_reward: {}\n".format(suc_rwd))
									f.write("slack_reward: -0.01\n")
									f.write("potential_reward_weight: {}\n".format(pot_rwd))
									f.write("collision_reward_weight: {}\n".format(col_rwd))
									f.write("collision_ignore_link_a_ids: []\n")
									f.write("collision_ignore_body_b_ids: [1,3]\n")
									f.write("max_collisions_allowed: 0\n\n")

									f.write("discount_factor: {}\n\n".format(gamma))

									f.write("dist_tol: {}\n".format(tol))
									f.write("max_step: {}\n\n".format(num_steps))

									f.write("output: [sensor, depth, rgb, seg]\n")
									f.write("resolution: 256\n")
									f.write("vertical_fov: 90\n\n")

									f.write("use_filler: true\n")
									f.write("display_ui: false\n")
									f.write("show_diagnostics: false\n")
									f.write("ui_num: 2\n")
									f.write("ui_components: [RGB_FILLED, DEPTH]\n\n")

									f.write("speed:\n")
									f.write("  timestep: 0.001\n")
									f.write("  frameskip: 10\n\n")

									f.write("mode: web_ui\n")
									f.write("verbose:\n")
									f.write("fast_lq_render: true\n\n")

									f.write("visual_object_at_initial_target_pos: true\n")
									f.write("target_visual_object_visible_to_agent: true\n\n")

									f.write("debug: false")
									f.close()
