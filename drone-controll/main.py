from drone_controller import ParrotAnafi
import asyncio

async def main():
    """Example of using the system"""
    # Initialize drone
    drone = ParrotAnafi("192.168.42.1")
    
    try:
        # Connect
        if await drone.connect():
            print("✅ Successfully connected!")
            
            # Show available missions
            missions = drone.get_available_missions()
            print(f"🎯 Available missions: {missions}")
            
            # Show mission info
            for mission_name in missions:
                info = drone.get_mission_info(mission_name)
                print(f"📋 {mission_name}: {info['description']}")
            
            # Execute Room Search mission
            print("\n🚀 Starting Room Search...")
            result = await drone.execute_mission("Room Search")
            print(f"📊 Result: {result}")
            
            # Execute Ring Challenge mission
            print("\n🎯 Starting Ring Challenge...")
            result = await drone.execute_mission("Ring Challenge")
            print(f"📊 Result: {result}")
            
        else:
            print("❌ Connection failed")
            
    except KeyboardInterrupt:
        print("\n⏹️ Manual stop")
    finally:
        # Disconnect
        await drone.disconnect()
        print("👋 Disconnected")

if __name__ == "__main__":
    asyncio.run(main())