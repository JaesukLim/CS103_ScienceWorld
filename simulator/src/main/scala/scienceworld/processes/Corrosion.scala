package scienceworld.processes

import scienceworld.objects.substance.Rust
import scienceworld.struct.EnvObject

object Corrosion {
  private val corrosiveSubstances = Set("salt water")

  def corrosionTick(obj:EnvObject): Unit = {
    if (obj.isDeleted() || obj.propMaterial.isEmpty) return
    if (!isCorrodable(obj)) return
    if (!isInDirectContactWithCorrosiveLiquid(obj)) return

    obj.propMaterial.get.corrosionTicks += 1
    if (obj.propMaterial.get.corrosionTicks < corrosionThreshold(obj)) return

    applyCorrosion(obj)
  }

  private def isCorrodable(obj:EnvObject): Boolean = {
    if (obj.name.toLowerCase.startsWith("corroded ")) return false
    if (obj.name.toLowerCase == "rust") return false
    if (obj.propMaterial.get.stateOfMatter != "solid") return false

    obj.name.toLowerCase match {
      case "iron block" => true
      case _ => obj.propMaterial.get.electricallyConductive
    }
  }

  private def isInDirectContactWithCorrosiveLiquid(obj:EnvObject): Boolean = {
    val siblings = obj.getContainer().map(_.getContainedObjects()).getOrElse(Set.empty[EnvObject])
    siblings.exists(other => (other != obj) && isCorrosiveLiquid(other)) ||
      obj.getContainedObjects().exists(isCorrosiveLiquid)
  }

  private def isCorrosiveLiquid(obj:EnvObject): Boolean = {
    if (obj.propMaterial.isEmpty) return false

    val material = obj.propMaterial.get
    corrosiveSubstances.contains(material.substanceName) &&
      material.stateOfMatter == "liquid"
  }

  private def corrosionThreshold(obj:EnvObject): Int = {
    obj.name.toLowerCase match {
      case "iron block" => 4
      case "wire" => 3
      case "battery" => 3
      case _ => 5
    }
  }

  private def applyCorrosion(obj:EnvObject): Unit = {
    if (obj.name.toLowerCase == "iron block") {
      replaceObject(obj, new Rust())
      return
    }

    obj.name = "corroded " + obj.name
    obj.propMaterial.get.color = "brown"
    obj.propMaterial.get.electricallyConductive = false

    obj.disconnectElectricalTerminals()

    if (obj.propDevice.isDefined) {
      obj.propDevice.get.isBroken = true
      obj.propDevice.get.isActivated = false
    }
  }

  private def replaceObject(oldObj:EnvObject, newObj:EnvObject): Unit = {
    val container = oldObj.getContainer()
    if (container.isEmpty) return

    if (oldObj.propMaterial.isDefined && newObj.propMaterial.isDefined) {
      newObj.propMaterial.get.temperatureC = oldObj.propMaterial.get.temperatureC
    }

    oldObj.delete()
    container.get.addObject(newObj)
  }
}
